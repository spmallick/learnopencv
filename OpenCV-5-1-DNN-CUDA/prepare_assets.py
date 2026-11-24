"""Export two pretrained ImageNet models and verify them against their PyTorch weights."""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
import torch
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16


# These weight identifiers name a specific training run. DEFAULT could change in torchvision.
MODELS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1,
                 "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1,
                 "c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9"),
}

# Both photos are CC0. Chelsea is by Stefan van der Walt, coffee by Rachel Michetti.
# Credits: https://scikit-image.org/docs/0.25.x/api/skimage.data.html
PHOTO_URL = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/"
PHOTOS = {
    "chelsea.png": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    "coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
}


def sha256(path):
    """Stream model files through the hash to avoid an extra copy of the weights."""
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def download(url, path, expected):
    """Only use cached downloads when their entire SHA-256 matches."""
    if path.is_file() and sha256(path) == expected:
        return
    # Keep a partial or unverified download out of both the cache and the benchmark.
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        temporary = Path(staging) / path.name
        urllib.request.urlretrieve(url, temporary)
        if sha256(temporary) != expected:
            raise ValueError(f"Checksum mismatch downloading {url}")
        temporary.replace(path)


def export_and_check(model, samples, reference, output, dynamic):
    # Batches may change, but channel count and pixel dimensions stay fixed for both models.
    axes = {"images": {0: "batch"}, "logits": {0: "batch"}} if dynamic else None
    # The traced attention graph contains arithmetic on known dimensions. Fold the same
    # way for both exports so static shapes can become constants for CUDA Graph capture.
    raw = output.with_suffix(".raw.onnx")
    torch.onnx.export(
        model, samples[:1], str(raw), input_names=["images"], output_names=["logits"],
        opset_version=17, dynamo=False, dynamic_axes=axes,
    )
    onnxslim.slim(str(raw), str(output))
    raw.unlink()
    onnx.checker.check_model(str(output))
    # Compare all 1,000 logits per image with eager PyTorch, including a real two-photo batch.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    max_error = 0.0
    slices = [slice(0, 1), slice(1, 2)]
    if dynamic:
        slices.append(slice(0, 2))
    for selection in slices:
        actual, = session.run(["logits"], {"images": samples[selection].numpy()})
        expected = reference[selection]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return max_error


def prepare(output_dir, cache_dir, model_names, shapes):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename, digest in PHOTOS.items():
        download(PHOTO_URL + filename, output_dir / filename, digest)
    labels = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    (output_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    # Emit standard attention operators that the ONNX runtimes can read.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(4)
    for name in model_names:
        constructor, weights, digest = MODELS[name]
        checkpoint = cache_dir / weights.url.rsplit("/", 1)[-1]
        download(weights.url, checkpoint, digest)
        model = constructor(weights=None).eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        # The export check uses the original torchvision recipe on two distinct RGB photographs.
        transformed = []
        for filename in PHOTOS:
            with Image.open(output_dir / filename) as photo:
                transformed.append(weights.transforms()(photo.convert("RGB")))
        samples = torch.stack(transformed)
        with torch.inference_mode():
            reference = model(samples).numpy()
        if weights.meta["categories"] != labels:
            raise ValueError(f"Class order for {name} differs from labels.txt")
        evidence = {
            "name": name, "weights_url": weights.url, "weights_sha256": digest,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "onnxslim": onnxslim.__version__, "opset": 17,
            "torchvision_top1": dict(zip(PHOTOS, (labels[i] for i in reference.argmax(1)))),
            "exports": {},
        }
        with tempfile.TemporaryDirectory(dir=output_dir) as staging:
            for shape in shapes:
                filename = f"{name}-{shape}.onnx"
                staged = Path(staging) / filename
                max_error = export_and_check(model, samples, reference, staged, shape == "dynamic")
                evidence["exports"][filename] = {
                    "sha256": sha256(staged), "max_abs_error_vs_torch": max_error,
                }
                staged.replace(output_dir / filename)
            print(json.dumps(evidence, indent=2), flush=True)
            (output_dir / f"{name}-export.json").write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/learnopencv/dnn51")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--fixed-only", action="store_true", help="export only the fixed batch-one models")
    args = parser.parse_args()
    prepare(args.output_dir, args.cache_dir, args.models, ("fixed",) if args.fixed_only else ("fixed", "dynamic"))

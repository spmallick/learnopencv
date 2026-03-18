import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn
from dataset import AttributesDataset, mean, std
from model import MultiOutputModel


class TritonExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output["color"], output["gender"], output["article"]


def checkpoint_load(model, checkpoint_path):
    print("Restoring checkpoint: {}".format(checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)


def write_label_file(path, labels):
    with open(path, "w", encoding="utf-8") as file:
        for label in labels:
            file.write("{}\n".format(label))


def build_config(model_name, attributes, max_batch_size):
    lines = [
        'name: "{}"'.format(model_name),
        'backend: "onnxruntime"',
        "max_batch_size: {}".format(max_batch_size),
        "",
        "input [",
        "  {",
        '    name: "input"',
        "    data_type: TYPE_FP32",
        "    dims: [ 3, -1, -1 ]",
        "  }",
        "]",
        "",
        "output [",
        "  {",
        '    name: "color"',
        "    data_type: TYPE_FP32",
        "    dims: [ {} ]".format(attributes.num_colors),
        '    label_filename: "color_labels.txt"',
        "  },",
        "  {",
        '    name: "gender"',
        "    data_type: TYPE_FP32",
        "    dims: [ {} ]".format(attributes.num_genders),
        '    label_filename: "gender_labels.txt"',
        "  },",
        "  {",
        '    name: "article"',
        "    data_type: TYPE_FP32",
        "    dims: [ {} ]".format(attributes.num_articles),
        '    label_filename: "article_labels.txt"',
        "  }",
        "]",
    ]

    if max_batch_size > 0:
        lines.extend(["", "dynamic_batching {}"])

    lines.append("")
    return "\n".join(lines)


def build_metadata(model_name, attributes, max_batch_size):
    return {
        "model_name": model_name,
        "max_batch_size": max_batch_size,
        "input": {
            "name": "input",
            "datatype": "FP32",
            "layout": "NCHW",
            "shape": ["batch", 3, "height", "width"],
            "normalization": {
                "mean": mean,
                "std": std,
            },
        },
        "outputs": {
            "color": {
                "num_classes": attributes.num_colors,
                "labels": attributes.color_labels.tolist(),
            },
            "gender": {
                "num_classes": attributes.num_genders,
                "labels": attributes.gender_labels.tolist(),
            },
            "article": {
                "num_classes": attributes.num_articles,
                "labels": attributes.article_labels.tolist(),
            },
        },
    }


def export_model(wrapper, output_path, height, width, opset):
    dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32)
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "color": {0: "batch_size"},
        "gender": {0: "batch_size"},
        "article": {0: "batch_size"},
    }
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["color", "gender", "article"],
        "dynamic_axes": dynamic_axes,
    }

    # PyTorch 2.6+ may default to the new dynamo exporter, which adds an
    # onnxscript dependency. Use the legacy exporter when the flag exists so
    # this helper keeps working in lighter environments.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.inference_mode():
        torch.onnx.export(wrapper, dummy_input, str(output_path), **export_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a trained multi-output classifier to an ONNX model repository layout for Triton."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained PyTorch checkpoint.")
    parser.add_argument(
        "--attributes_file",
        type=str,
        default="./fashion-product-images/styles.csv",
        help="Path to the CSV file used to build label mappings.",
    )
    parser.add_argument(
        "--model_repository",
        type=str,
        default="./triton_model_repository",
        help="Root folder of the Triton model repository to create.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="fashion_multi_output",
        help="Model name to use inside the Triton repository.",
    )
    parser.add_argument(
        "--model_version",
        type=int,
        default=1,
        help="Numeric model version directory to create under the model name.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Triton max_batch_size to place into config.pbtxt.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Dummy input height used during ONNX export. Runtime height stays dynamic.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Dummy input width used during ONNX export. Runtime width stays dynamic.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    args = parser.parse_args()

    if args.max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1 for this exported Triton config.")

    attributes = AttributesDataset(args.attributes_file)
    model = MultiOutputModel(
        n_color_classes=attributes.num_colors,
        n_gender_classes=attributes.num_genders,
        n_article_classes=attributes.num_articles,
    )
    checkpoint_load(model, args.checkpoint)
    model.eval()

    wrapper = TritonExportWrapper(model)
    wrapper.eval()

    model_dir = Path(args.model_repository).expanduser().resolve() / args.model_name
    version_dir = model_dir / str(args.model_version)
    version_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = version_dir / "model.onnx"
    export_model(wrapper, onnx_path, args.height, args.width, args.opset)

    config_path = model_dir / "config.pbtxt"
    config_path.write_text(
        build_config(args.model_name, attributes, args.max_batch_size),
        encoding="utf-8",
    )

    write_label_file(model_dir / "color_labels.txt", attributes.color_labels.tolist())
    write_label_file(model_dir / "gender_labels.txt", attributes.gender_labels.tolist())
    write_label_file(model_dir / "article_labels.txt", attributes.article_labels.tolist())

    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(build_metadata(args.model_name, attributes, args.max_batch_size), indent=2),
        encoding="utf-8",
    )

    print("Exported Triton model repository to {}".format(model_dir))
    print("  ONNX model: {}".format(onnx_path))
    print("  Config: {}".format(config_path))
    print("  Metadata: {}".format(metadata_path))

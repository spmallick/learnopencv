"""
Export YOLO26 to ONNX for OpenCV 5 DNN inference.

We export a plain detection head with a STATIC input shape and opset 12,
which is what the OpenCV 5 DNN engine expects. We deliberately do NOT embed
NMS / end2end ops in the graph, so the ONNX file stays portable.

The exported .onnx files go into the repo-root `models/` folder (the same place
detect_*/benchmark expect them). Ultralytics downloads the yolo26n.pt weights
automatically. The exported file is size-tagged (e.g. yolo26n_640.onnx,
yolo26n_1280.onnx) so multiple input sizes coexist.

Usage:  python export_yolo26_onnx.py --imgsz 640
"""
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

# scripts/ -> repo root -> models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolo26n.pt"   # nano = fastest on CPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640,
                    help="fixed square input size (e.g. 640 fast, 1280 for small objects)")
    args = ap.parse_args()

    weights = MODELS_DIR / MODEL_NAME
    # Ultralytics downloads the weights automatically if missing. Move the
    # downloaded file into models/ after export so the project root stays clean.
    model = YOLO(str(weights) if weights.exists() else MODEL_NAME)

    onnx_path = model.export(
        format="onnx",
        opset=12,          # OpenCV 5 DNN needs opset >= 12
        imgsz=args.imgsz,  # fixed square input
        dynamic=False,     # STATIC shapes (avoids the DNN Concat-parse issue)
        simplify=True,     # fold constants for a cleaner graph
        nms=False,         # keep the raw detection head; we run NMS ourselves
    )
    onnx_path = Path(onnx_path)

    # Give the file a size-tagged name (in models/) so 640 and 1280 coexist.
    tagged = MODELS_DIR / f"yolo26n_{args.imgsz}.onnx"
    if onnx_path.resolve() != tagged.resolve():
        shutil.move(onnx_path, tagged)
    downloaded_weights = Path(MODEL_NAME)
    if not weights.exists() and downloaded_weights.exists():
        shutil.move(downloaded_weights, weights)
    print("Exported ONNX:", tagged)
    onnx_path = tagged

    # Inspect the ONNX I/O so we know exactly how to parse it in OpenCV.
    import onnx
    m = onnx.load(str(onnx_path))

    def shape_of(vi):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
        return dims

    print("\n--- ONNX inputs ---")
    for vi in m.graph.input:
        print(" ", vi.name, shape_of(vi))
    print("--- ONNX outputs ---")
    for vi in m.graph.output:
        print(" ", vi.name, shape_of(vi))


if __name__ == "__main__":
    main()

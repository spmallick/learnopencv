"""
Export YOLO26 segmentation and pose variants to ONNX for OpenCV 5 DNN.
Static shape, opset 12, no embedded NMS (same recipe as Part 1's detector).
Prints the ONNX input/output tensor shapes so we know how to parse them in C++.
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(exist_ok=True)

VARIANTS = [
    ("yolo26n-seg.pt", "seg"),
    ("yolo26n-pose.pt", "pose"),
]
IMGSZ = 640


def shapes(onnx_path):
    import onnx
    m = onnx.load(str(onnx_path))

    def sh(vi):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
        return dims
    print("  inputs :", [(vi.name, sh(vi)) for vi in m.graph.input])
    print("  outputs:", [(vi.name, sh(vi)) for vi in m.graph.output])


def main():
    for weights, tag in VARIANTS:
        print(f"\n=== {weights} ===")
        model_path = MODELS / weights
        model = YOLO(str(model_path) if model_path.exists() else weights)
        out = model.export(format="onnx", opset=12, imgsz=IMGSZ,
                           dynamic=False, simplify=True, nms=False)
        out = Path(out)
        tagged = MODELS / f"yolo26n_{tag}_{IMGSZ}.onnx"
        if out.resolve() != tagged.resolve():
            shutil.move(out, tagged)
        downloaded_weights = Path(weights)
        if not model_path.exists() and downloaded_weights.exists():
            shutil.move(downloaded_weights, model_path)
        print("exported:", tagged)
        shapes(tagged)


if __name__ == "__main__":
    main()

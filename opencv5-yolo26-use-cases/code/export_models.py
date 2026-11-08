"""
export_models.py
Export YOLO26 nano to ONNX at 640 and 960 input size into ../models.

This is the only step that needs Ultralytics (and PyTorch). Everything else in this
repository runs on OpenCV 5 + NumPy. The settings are the ones the OpenCV 5 importer
accepts: opset 12, a fixed input size, no exported NMS (YOLO26 is NMS-free anyway).
"""

import os
import shutil

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")

if __name__ == "__main__":
    os.makedirs(MODELS, exist_ok=True)
    for size in (640, 960):
        dst = os.path.join(MODELS, f"yolo26n_{size}.onnx")
        if os.path.exists(dst):
            print("have", dst)
            continue
        path = YOLO("yolo26n.pt").export(format="onnx", opset=12, imgsz=size, dynamic=False, nms=False, simplify=True)
        shutil.move(path, dst)
        print("wrote", dst)

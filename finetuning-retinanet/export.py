import torch
import os
from model import create_model
from config import NUM_CLASSES, DEVICE

"""
Usage:
    python export.py
This will produce 'retinanet.onnx' in the 'outputs/' directory.
Make sure 'outputs/best_model.pth' exists.
"""


def export_onnx_model(onnx_path="outputs/retinanet.onnx", input_size=640):
    # 1. Create the model and load weights.
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load("outputs/best_model_79.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # 2. Create a dummy input of the correct size.
    #    RetinaNet typically expects (N, 3, H, W). Here we use 1x3x640x640.
    dummy_input = torch.randn(1, 3, input_size, input_size, device=DEVICE)

    # 3. Export the model to ONNX format.
    #    For object detection, we often need 'opset_version >= 11'.
    torch.onnx.export(
        model,  # model
        dummy_input,  # input
        onnx_path,  # where to save
        input_names=["images"],  # name the input tensor
        output_names=["boxes", "scores", "labels"],  # typical RetinaNet raw outputs
        opset_version=11,  # or higher
        do_constant_folding=True,  # fold constant ops
        export_params=True,  # store the trained parameter weights
    )
    print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    export_onnx_model()

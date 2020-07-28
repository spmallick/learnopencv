import argparse
import platform

import coremltools
import numpy as np
import onnx
import onnxruntime
import torch
import torch.onnx
import torchvision
from onnx_coreml import convert
from onnxsim import simplify


def check_onnx_output(filename, input_data, torch_output):
    session = onnxruntime.InferenceSession(filename)
    input_name = session.get_inputs()[0].name
    result = session.run([], {input_name: input_data.numpy()})
    for test_result, gold_result in zip(result, torch_output.values()):
        np.testing.assert_almost_equal(
            gold_result.cpu().numpy(), test_result, decimal=3,
        )
    return result


def check_onnx_model(model, onnx_filename, input_image):
    with torch.no_grad():
        torch_out = {"output": model(input_image)}
    check_onnx_output(onnx_filename, input_image, torch_out)
    print("PyTorch output and ONNX output values are equal")
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    print("The model is checked!")
    return onnx_model


def check_coreml_model(coreml_filename, torch_model, input_data):

    # get PyTorch model output
    with torch.no_grad():
        torch_output = {"output": torch_model(input_data)}

    # get CoreML model output
    coreml_model = coremltools.models.MLModel(coreml_filename, useCPUOnly=True)

    # convert input to numpy and get coreml model prediction
    input_data = input_data.cpu().numpy()
    pred = coreml_model.predict({"input": input_data})

    for key in pred:
        np.testing.assert_almost_equal(
            torch_output[key].cpu().numpy(), pred[key], decimal=3,
        )
    print("CoreML model is checked!")
    return pred


def save_onnx_from_torch(
    model, model_name, input_image, input_names=None, output_names=None, simplify=False,
):
    # Section 1: PyTorch model conversion --
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # set mode to evaluation and change device to cpu
    model.eval()
    model.cpu()
    onnx_filename = model_name + ".onnx"

    # export our model to ONNX format
    torch.onnx.export(
        model,
        input_image,
        onnx_filename,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )

    # Section 2: Model testing
    onnx_model = check_onnx_model(model, onnx_filename, input_image)

    # Section 3: ONNX simplifier
    if simplify:
        filename = model_name + "_simplified.onnx"
        onnx_model_simplified, check = simplify_onnx(onnx_model, filename)
        onnx.checker.check_model(onnx_model_simplified)
        check_onnx_model(model, filename, input_image)
        return onnx_model_simplified
    else:
        return onnx_model


def simplify_onnx(onnx_model, filename):
    simplified_model, check = simplify(onnx_model)
    onnx.save_model(simplified_model, filename)
    return simplified_model, check


def convert_onnx_to_coreml(onnx_model, model_name, torch_model, input_data):
    model_coreml = convert(onnx_model, minimum_ios_deployment_target="13")
    coreml_filename = model_name + ".mlmodel"
    model_coreml.save(coreml_filename)
    if platform.system() == "Darwin":
        check_coreml_model(coreml_filename, torch_model, input_data)
    return model_coreml


def main(args):
    state = 42
    np.random.seed(state)
    torch.manual_seed(state)

    print("Start model conversion")
    model_name = args.model_name
    input_size = args.input_size
    simp = args.simplify

    # get model from torchvision
    torch_model = getattr(torchvision.models, model_name)(pretrained=True)

    # random image to make a network forward pass
    dummy_input = torch.randn(1, 3, input_size, input_size, device="cpu")

    # save ONNX model
    onnx_model = save_onnx_from_torch(
        torch_model, model_name, input_image=dummy_input, simplify=simp,
    )
    convert_onnx_to_coreml(
        onnx_model, model_name, torch_model=torch_model, input_data=dummy_input,
    )

    print("PyTorch model has been converted to CoreML format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-mn",
        required=True,
        type=str,
        help="model name from torchvision.model",
    )
    parser.add_argument(
        "--simplify",
        "-s",
        action="store_true",
        help="Simplify ONNX model using onnx-simplifier",
    )
    parser.add_argument("--input_size", default=224, type=int, help="input image size")
    args = parser.parse_args()
    main(args)

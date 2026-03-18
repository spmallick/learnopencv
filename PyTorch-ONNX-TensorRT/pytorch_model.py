import albumentations as A
import cv2
import onnx
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.models import ResNet50_Weights, resnet50


def get_runtime_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        return torch.device("cpu")
    return torch.device("cuda")


def preprocess_image(img_path):
    transforms = A.Compose(
        [
            A.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    )

    input_img = cv2.imread(img_path)
    input_data = transforms(image=input_img)["image"]
    return torch.unsqueeze(input_data, 0)


def postprocess(output_data):
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    while confidences[indices[0][i]] > 0.5:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1


def main():
    device = get_runtime_device()

    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    input_tensor = preprocess_image("turkish_coffee.jpg").to(device)

    model.eval()
    output = model(input_tensor)
    postprocess(output)

    onnx_file_path = "resnet50.onnx"
    torch.onnx.export(
        model,
        input_tensor,
        onnx_file_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
    )

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", onnx_file_path)


if __name__ == "__main__":
    main()

import cv2
import onnx
import torch
from albumentations import (
    Compose,
    Resize,
)
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensor
from torchvision import models

# load pre-trained model ------------------------------------------------------
model = models.resnet50(pretrained=True)

# preprocessing stage ---------------------------------------------------------
# transformations for the input data
transforms = Compose(
    [
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ],
)

# read input image
input_img = cv2.imread("turkish_coffee.jpg")
# do transformations
input_data = transforms(image=input_img)["image"]
# prepare batch
batch_data = torch.unsqueeze(input_data, 0).cuda()

# inference stage -------------------------------------------------------------
model.eval()
model.cuda()
output_data = model(batch_data)

# post-processing stage -------------------------------------------------------
# get class names
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]
# calculate human-readable value by softmax
confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
# find top predicted classes
_, indices = torch.sort(output_data, descending=True)
i = 0
# print the top classes predicted by the model
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

# convert to ONNX -------------------------------------------------------------
onnx_filename = "resnet50.onnx"
torch.onnx.export(
    model,
    batch_data,
    onnx_filename,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
)

onnx_model = onnx.load(onnx_filename)
# check that the model converted fine
onnx.checker.check_model(onnx_model)

print("Model was successfully converted to ONNX format.")
print("It was saved to", onnx_filename)

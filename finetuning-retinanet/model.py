import torchvision
import torch

from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from config import NUM_CLASSES


def create_model(num_classes=91):
    """
    Creates a RetinaNet-ResNet50-FPN v2 model pre-trained on COCO.
    Replaces the classification head for the required number of classes.
    """
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    num_anchors = model.head.classification_head.num_anchors

    # Replace the classification head
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256, num_anchors=num_anchors, num_classes=num_classes, norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    return model


if __name__ == "__main__":
    model = create_model(num_classes=NUM_CLASSES)
    print(model)
    # Total parameters:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    # Trainable parameters:
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    augmentations_train = A.Compose(
        [
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    )
    return lambda img: augmentations_train(image=np.array(img))


def get_test_augmentation():
    augmentations_val = A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    )
    return lambda img: augmentations_val(image=np.array(img))


def unnormalize(tensor):
    for channel, mean, std in zip(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)):
        channel.mul_(std).add_(mean)
    return tensor

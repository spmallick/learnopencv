import os
import argparse
import torch
import numpy as np
from torchvision import transforms
import cv2

IMG_EXT = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')

CLASS_MAP = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
             "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
             "potted plant": 16, "sheep": 17, "sofa": 18, "train": 19, "tv/monitor": 20}


def trimap(probs, size, conf_threshold):
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [3]: an image with probabilities of each pixel being the foreground, size of dilation kernel,
    foreground confidence threshold
    Output    : a trimap
    """
    mask = (probs > 0.05).astype(np.uint8) * 255

    pixels = 2 * size + 1
    kernel = np.ones((pixels, pixels), np.uint8)

    dilation = cv2.dilate(mask, kernel, iterations=1)

    remake = np.zeros_like(mask)
    remake[dilation == 255] = 127  # Set every pixel within dilated region as probably foreground.
    remake[probs > conf_threshold] = 255  # Set every pixel with large enough probability as definitely foreground.

    return remake


def parse_args():
    parser = argparse.ArgumentParser(description="Deeplab Segmentation")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory to save the output results. (required)",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default='person',
        choices=CLASS_MAP.keys(),
        help="Type of the foreground object.",
    )
    parser.add_argument(
        "--show",
        action='store_true',
        help="Use to show results.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default='0.95',
        help="Confidence threshold for the foreground object. "
             "You can play with it to get better looking trimaps.",
    )

    args = parser.parse_args()
    return args


def main(input_dir, target_class, show, conf_threshold):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    trimaps_path = os.path.join(input_dir, "trimaps")
    os.makedirs(trimaps_path, exist_ok=True)

    images_list = os.listdir(input_dir)
    for filename in images_list:
        if not filename.endswith(IMG_EXT):
            continue
        input_image = cv2.imread(os.path.join(input_dir, filename))
        original_image = input_image.copy()

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)['out'][0]
            output = torch.softmax(output, 0)

        output_cat = output[CLASS_MAP[target_class], ...].numpy()

        trimap_image = trimap(output_cat, 7, conf_threshold)
        trimap_filename = os.path.basename(filename).split('.')[0] + '.png'
        cv2.imwrite(os.path.join(trimaps_path, trimap_filename), trimap_image)

        if show:
            cv2.imshow('mask', output_cat)
            cv2.imshow('image', original_image)
            cv2.imshow('trimap', trimap_image)
            cv2.waitKey(0)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.target_class, args.show, args.conf_threshold)

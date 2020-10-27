import argparse
import os

import cv2
import numpy as np
import torch
from dataloader import PredDataset
from networks.models import build_model
from networks.transforms import (
    groupnorm_normalise_image,
    trimap_transform,
)
from tqdm import tqdm


def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float()


def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    """ Scales inputs to multiple of 8. """
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def swap_bg(image, alpha):
    green_bg = np.zeros_like(image).astype(np.float32)
    green_bg[:, :, 1] = 255

    alpha = alpha[:, :, np.newaxis]
    result = alpha * image.astype(np.float32) + (1 - alpha) * green_bg
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def predict_fba_folder(model, args):
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    dataset_test = PredDataset(args.image_dir, args.trimap_dir)

    gen = iter(dataset_test)
    for item_dict in tqdm(gen):
        image_np = item_dict["image"]
        trimap_np = item_dict["trimap"]

        fg, bg, alpha = pred(image_np, trimap_np, model, args)

        cv2.imwrite(
            os.path.join(save_dir, item_dict["name"][:-4] + "_fg.png"),
            fg[:, :, ::-1] * 255,
        )
        cv2.imwrite(
            os.path.join(save_dir, item_dict["name"][:-4] + "_bg.png"),
            bg[:, :, ::-1] * 255,
        )
        cv2.imwrite(
            os.path.join(save_dir, item_dict["name"][:-4] + "_alpha.png"), alpha * 255,
        )

        example_swap_bg = swap_bg(fg[:, :, ::-1] * 255, alpha)
        cv2.imwrite(
            os.path.join(save_dir, item_dict["name"][:-4] + "_swapped_bg.png"), example_swap_bg,
        )


def pred(image_np: np.ndarray, trimap_np: np.ndarray, model, args) -> np.ndarray:
    """ Predict alpha, foreground and background.
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    """
    h, w = trimap_np.shape[:2]

    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():

        image_torch = np_to_torch(image_scale_np).to(args.device)
        trimap_torch = np_to_torch(trimap_scale_np).to(args.device)

        trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np)).to(
            args.device,
        )
        image_transformed_torch = groupnorm_normalise_image(
            image_torch.clone(), format="nchw",
        )

        output = model(
            image_torch,
            trimap_torch,
            image_transformed_torch,
            trimap_transformed_torch,
        )

        output = cv2.resize(
            output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4,
        )
    alpha = output[:, :, 0]
    fg = output[:, :, 1:4]
    bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]
    return fg, bg, alpha


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument("--encoder", default="resnet50_GN_WS", help="Encoder model")
    parser.add_argument("--decoder", default="fba_decoder", help="Decoder model")
    parser.add_argument("--weights", default="FBA.pth")
    parser.add_argument("--image_dir", default="./examples/images", help="")
    parser.add_argument(
        "--trimap_dir", default="./examples/trimaps", help="",
    )
    parser.add_argument("--output_dir", default="./examples/predictions", help="")
    parser.add_argument("--device", default="cpu", help="Device for inference on")

    args = parser.parse_args()
    model = build_model(args).to(args.device)
    model.eval()
    predict_fba_folder(model, args)

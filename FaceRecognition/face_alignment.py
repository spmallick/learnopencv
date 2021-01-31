# Original code
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/align/face_align.py

import argparse
import os

import numpy as np
from align.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from align.detector import detect_faces
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tags",
        help="specify your tags for raw datasets",
        default="test",
        nargs='+',
        required=True
    )
    parser.add_argument(
        "--crop_size",
        help="specify size of aligned faces",
        default=112,
        choices=[112, 224],
        type=int,
    )
    args = parser.parse_args()

    tags = args.tags
    crop_size = args.crop_size
    scale = crop_size / 112.0
    reference = get_reference_facial_points(default_square=True) * scale

    for tag in tags:
        source_root = os.path.join("data", tag)
        dest_root = source_root + "_aligned"
        if not os.path.isdir(dest_root):
            os.mkdir(dest_root)

        for subfolder in tqdm(os.listdir(source_root)):
            if not os.path.isdir(os.path.join(dest_root, subfolder)):
                os.mkdir(os.path.join(dest_root, subfolder))
            for image_name in os.listdir(os.path.join(source_root, subfolder)):
                print(
                    "Processing\t{}".format(
                        os.path.join(source_root, subfolder, image_name),
                    ),
                )
                img = Image.open(os.path.join(source_root, subfolder, image_name))
                try:  # Handle exception
                    _, landmarks = detect_faces(img)
                except Exception:
                    print(
                        "{} is discarded due to exception!".format(
                            os.path.join(source_root, subfolder, image_name),
                        ),
                    )
                    continue
                if (
                    len(landmarks) == 0
                ):  # If the landmarks cannot be detected, the img will be discarded
                    print(
                        "{} is discarded due to non-detected landmarks!".format(
                            os.path.join(source_root, subfolder, image_name),
                        ),
                    )
                    continue
                facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
                warped_face = warp_and_crop_face(
                    np.array(img),
                    facial5points,
                    reference,
                    crop_size=(crop_size, crop_size),
                )
                img_warped = Image.fromarray(warped_face)
                if image_name.split(".")[-1].lower() not in ["jpg", "jpeg"]:
                    image_name = ".".join(image_name.split(".")[:-1]) + ".jpg"
                img_warped.save(os.path.join(dest_root, subfolder, image_name))

import os
from argparse import ArgumentParser

import cv2
import lmdb
import numpy as np

from tools import get_images_paths


def store_many_lmdb(images_list, save_path):

    num_images = len(images_list)  # number of images in our folder

    file_sizes = [os.path.getsize(item) for item in images_list]  # all file sizes
    max_size_index = np.argmax(file_sizes)  # the maximum file size index

    # maximum database size in bytes
    map_size = num_images * cv2.imread(images_list[max_size_index]).nbytes * 10

    env = lmdb.open(save_path, map_size=map_size)  # create lmdb environment

    with env.begin(write=True) as txn:  # start writing to environment
        for i, image in enumerate(images_list):
            with open(image, "rb") as file:
                data = file.read()  # read image as bytes
                key = f"{i:08}"  # get image key
                txn.put(key.encode("ascii"), data)  # put the key-value into database

    env.close()  # close the environment


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="path to the images folder to collect",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help='path to the output environment directory file i.e. "path/to/folder/env/"',
    )

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    images = get_images_paths(args.path)
    store_many_lmdb(images, args.output)

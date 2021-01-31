import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from embeddings import get_embeddings
from sklearn.manifold import TSNE
from tqdm import tqdm


def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = np.max(x) - np.min(x)

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image, x, y in tqdm(
        zip(images, tx, ty), desc="Building the T-SNE plot", total=len(images),
    ):

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(
            image, x, y, image_centers_area_size, offset,
        )

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.figure('Embeddings projection with t-SNE')
    plt.axis('off')
    plt.imshow(tsne_plot[:, :, ::-1])
    plt.savefig('tsne_plot.jpg')
    plt.show()


def visualize_tsne(tsne, images, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as images
    visualize_tsne_images(
        tx, ty, images, plot_size=plot_size, max_image_size=max_image_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tags",
        help="specify your tags for aligned faces datasets",
        default="test",
        nargs='+',
        required=True
    )
    args = parser.parse_args()
    tags = args.tags
    parser.add_argument(
        "--input_size",
        help="specify size of aligned faces, align and crop with padding",
        default=112,
        choices=[112, 224],
        type=int,
    )
    args = parser.parse_args()

    fix_random_seeds()

    # predict embeddings for each small dataset
    all_images = []
    all_embeddings = []
    tags = args.tags
    input_size = args.input_size
    for tag in tags:
        images, embeddings = get_embeddings(
            data_root=f"data/{tag}_aligned",
            model_root="checkpoint/backbone_ir50_ms1m_epoch120.pth",
            input_size=[input_size, input_size],
        )
        all_images.append(images)
        all_embeddings.append(embeddings)

    # concatenate images and embeddings from different persons into matrices
    all_images = np.vstack(all_images)
    all_embeddings = np.vstack(all_embeddings)
    # perplexity is lower than default, because number of samples is small
    tsne_results = TSNE(n_components=2, perplexity=8).fit_transform(all_embeddings)
    visualize_tsne(tsne_results, all_images, plot_size=1000, max_image_size=112)


if __name__ == "__main__":
    fix_random_seeds()
    main()

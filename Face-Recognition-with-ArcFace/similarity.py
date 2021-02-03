import argparse
import os
import cv2
import numpy as np
from embeddings import get_embeddings


def visualize_similarity(tag, input_size=[112, 112]):
    images, embeddings = get_embeddings(
        data_root=f"data/{tag}_aligned",
        model_root="checkpoint/backbone_ir50_ms1m_epoch120.pth",
        input_size=input_size,
    )

    # calculate cosine similarity matrix
    cos_similarity = np.dot(embeddings, embeddings.T)
    cos_similarity = cos_similarity.clip(min=0, max=1)
    # plot colorful grid from pair distance values in similarity matrix
    similarity_grid = plot_similarity_grid(cos_similarity, input_size)

    # pad similarity grid with images of faces
    horizontal_grid = np.hstack(images)
    vertical_grid = np.vstack(images)
    zeros = np.zeros((*input_size, 3))
    vertical_grid = np.vstack((zeros, vertical_grid))
    result = np.vstack((horizontal_grid, similarity_grid))
    result = np.hstack((vertical_grid, result))

    if not os.path.isdir("images"):
        os.mkdir("images")

    cv2.imwrite(f"images/{tag}.jpg", result)


def plot_similarity_grid(cos_similarity, input_size):
    n = len(cos_similarity)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            # create small colorful image from value in distance matrix
            value = cos_similarity[i][j]
            cell = np.empty(input_size)
            cell.fill(value)
            cell = (cell * 255).astype(np.uint8)
            # color depends on value: blue is closer to 0, green is closer to 1
            img = cv2.applyColorMap(cell, cv2.COLORMAP_WINTER)

            # add distance value as text centered on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{value:.2f}"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (img.shape[1] - textsize[0]) // 2
            text_y = (img.shape[0] + textsize[1]) // 2
            cv2.putText(
                img, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA,
            )
            row.append(img)
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows)
    return grid


if __name__ == "__main__":
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

    for tag in tags:            
        visualize_similarity(tag)

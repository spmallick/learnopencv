import argparse
import os
import os.path as osp
from shutil import copyfile

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Separate Food-101 into train/test folders",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        type=str,
        help="Path to root folder of the dataset",
    )
    args = parser.parse_args()
    classes = [
        "apple_pie",
        "bruschetta",
        "caesar_salad",
        "steak",
        "spring_rolls",
        "spaghetti_carbonara",
        "frozen_yogurt",
        "falafel",
        "mussels",
        "ramen",
        "onion_rings",
        "oysters",
        "risotto",
        "waffles",
        "cup_cakes",
        "grilled_cheese_sandwich",
        "fried_calamari",
        "huevos_rancheros",
        "croque_madame",
        "bread_pudding",
        "dumplings",
    ]
    assert osp.isdir(args.data_root)
    assert "images" in os.listdir(args.data_root)
    assert "meta" in os.listdir(args.data_root)
    os.makedirs(osp.join(args.data_root, "train"), exist_ok=True)
    os.makedirs(osp.join(args.data_root, "test"), exist_ok=True)
    for cls_name in classes:
        os.makedirs(osp.join(args.data_root, "train", cls_name), exist_ok=True)
        os.makedirs(osp.join(args.data_root, "test", cls_name), exist_ok=True)
    with open(osp.join(args.data_root, "meta", "train.txt"), "r") as file:
        for image in tqdm(file):
            image = image.rstrip()
            if image.split("/")[0] in classes:
                copyfile(
                    osp.join(args.data_root, "images", image + ".jpg"),
                    osp.join(args.data_root, "train", image + ".jpg"),
                )
    with open(osp.join(args.data_root, "meta", "test.txt"), "r") as file:
        for image in tqdm(file):
            image = image.rstrip()
            if image.split("/")[0] in classes:
                copyfile(
                    osp.join(args.data_root, "images", image + ".jpg"),
                    osp.join(args.data_root, "test", image + ".jpg"),
                )


if __name__ == "__main__":
    main()

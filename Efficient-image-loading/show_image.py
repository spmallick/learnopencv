from argparse import ArgumentParser

import cv2

from loader import (
    CV2Loader,
    LmdbLoader,
    PILLoader,
    TFRecordsLoader,
    TurboJpegLoader,
    methods,
)


def show_image(method, image):
    cv2.imshow(f"{method} image", image)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:  # check ESC pressing
        return True
    else:
        return False


def show_images(loader):
    num_images = len(loader)
    loader = iter(loader)
    for idx in range(num_images):
        image, time = next(loader)
        print_info(image, time)
        stop = show_image(type(loader).__name__, image)
        if stop:
            cv2.destroyAllWindows()
            return


def print_info(image, time):
    print(
        f"Image with {image.shape[0]}x{image.shape[1]} size has been loading for {time} seconds",
    )


def demo(method, path):
    loader = methods[method](path)  # get the image loader
    show_images(loader)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="path to image, folder of images, lmdb environment path or tfrecords database path",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["cv2", "pil", "turbojpeg", "lmdb", "tfrecords"],
        help="Image loading methods to use in benchmark",
    )

    args = parser.parse_args()

    demo(args.method, args.path)

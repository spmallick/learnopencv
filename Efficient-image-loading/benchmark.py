from argparse import ArgumentParser

import numpy as np
from prettytable import PrettyTable

from create_lmdb import store_many_lmdb
from create_tfrecords import store_many_tfrecords
from loader import (
    CV2Loader,
    LmdbLoader,
    PILLoader,
    TFRecordsLoader,
    TurboJpegLoader,
    methods,
)
from tools import get_images_paths


def count_time(loader, iters):
    time_list = []
    num_images = len(loader)
    for i in range(iters):
        loader = iter(loader)
        for idx in range(num_images):
            image, time = next(loader)
            time_list.append(time)
    time_list = np.asarray(time_list)
    print_stats(time_list, type(loader).__name__)
    return np.asarray(time_list)


def print_stats(time, name):
    print("Time measures for {}:".format(name))
    print("{} mean time - {:.8f} seconds".format(name, time.mean()))
    print("{} median time - {:.8f} seconds".format(name, np.median(time)))
    print("{} std time - {:.8f} seconds".format(name, time.std()))
    print("{} min time - {:.8f} seconds".format(name, time.min()))
    print("{} max time - {:.8f} seconds".format(name, time.max()))
    print("\n")


def benchmark(method, path, iters=100, **kwargs):

    image_loader = methods[method](path, **kwargs)  # get image loader
    time = count_time(image_loader, iters)  # measure the time for loading

    return time


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--path", "-p", type=str, help="path to image folder",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        required=True,
        choices=["cv2", "pil", "turbojpeg", "lmdb", "tfrecords"],
        help="Image loading methods to use in benchmark",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        required=True,
        choices=["BGR", "RGB"],
        help="Image color mode",
    )
    parser.add_argument(
        "--iters", type=int, help="Number of iterations to average the results",
    )
    args = parser.parse_args()

    benchmark_methods = args.method
    image_paths = get_images_paths(args.path)

    results = {}
    for method in benchmark_methods:
        if method == "lmdb":
            path = "./lmdb/images"
            store_many_lmdb(image_paths, path)
        elif method == "tfrecords":
            path = "./tfrecords/images.tfrecords"
            store_many_tfrecords(image_paths, path)
        else:
            path = args.path

        time = benchmark(method, path, mode=args.mode, iters=args.iters)
        results.update({method: time})

    table = PrettyTable(["Loader", "Mean time", "Median time"])

    print(
        f"Benchmark on {len(image_paths)} {args.mode} images with {args.iters} averaging iteration results:\n",
    )

    for method, time in results.items():
        table.add_row([method, time.mean(), np.median(time)])
    print(table)

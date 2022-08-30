import os
import cv2
import time
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count


def creat_dirs(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def chunk(length, n):
    for i in range(0, length, n):
        yield (i, i + n)


def ResizeWithAspectRatio(curr_dim, resize_to: int = 320):
    """returns new h and new w which maintains the aspect ratio"""

    h, w = curr_dim

    if h > w:
        r = resize_to / float(h)
        size = (int(w * r), resize_to)
    else:
        r = resize_to / float(w)
        size = (resize_to, int(h * r))
    return size[::-1]


def operation(data=None):

    prc_id = data["id"]
    images_paths = data["images_paths"]
    params = data["params"]

    SRC_DIR = params["SRC_DIR"]
    DST_DIR = params["DST_DIR"]
    IMG_SIZE = params["IMG_SIZE"]

    print(f"[INFO] starting process {prc_id}")

    for image_name in images_paths:

        src_image_path = os.path.join(SRC_DIR, image_name)
        dst_image_path = os.path.join(DST_DIR, image_name)

        image_true = cv2.imread(src_image_path, cv2.IMREAD_COLOR)

        imH, imW = image_true.shape[:2]

        if not (max(imW, imH) < IMG_SIZE):
            asp_h, asp_w = ResizeWithAspectRatio(curr_dim=(imH, imW), resize_to=IMG_SIZE)
            image_true = cv2.resize(image_true, (asp_w, asp_h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(dst_image_path, image_true)

    print(f"[INFO] finishing process {prc_id}")


def transform_images_xmls(source_dir, dest_dir, image_size=320):
    SRC_DIR = source_dir
    DST_DIR = dest_dir
    IMG_SIZE = image_size
    images_paths = []

    IMAGE_FORMATS = (".jpeg", ".JPEG", ".png", ".PNG", ".jpg", ".JPG")

    creat_dirs(SRC_DIR)
    creat_dirs(DST_DIR)

    params = {
        "SRC_DIR": SRC_DIR,
        "DST_DIR": DST_DIR,
        "IMG_SIZE": IMG_SIZE,
    }

    images_paths = [i for i in os.listdir(SRC_DIR) if i.endswith(IMAGE_FORMATS)]

    length = len(images_paths)
    procs = cpu_count()
    # procIDs = list(range(procs))
    print("Procs:", procs)

    numImagesPerProc = length / procs
    numImagesPerProc = int(np.ceil(numImagesPerProc))
    print("numImagesPerProc:", numImagesPerProc)

    chunked_image_paths = []

    for start, end in chunk(length, numImagesPerProc):
        chunked_image_paths.append(images_paths[start:end])

    payloads = []

    # loop over the set chunked image paths
    for i, imagePaths in enumerate(chunked_image_paths):
        data = {"id": i, "images_paths": imagePaths, "params": params}
        payloads.append(data)

    print("[INFO] Directory:", SRC_DIR)
    print("[INFO] Total images:", length)

    print(f"[INFO] launching pool using {procs} processes")
    pool = Pool(processes=procs)
    pool.map(operation, payloads)
    # close the pool and wait for all processes to finish
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")


def main():
    parser = argparse.ArgumentParser(description="Create Training and Validatin splits")
    parser.add_argument("-s", "--source-dir", required=True, type=str, help="Input Source folder path")
    parser.add_argument("-d", "--destination-dir", required=True, type=str, help="Output destination folder path")
    parser.add_argument("-x", "--img-size", required=True, type=int, help="size of resized Image ")

    args = parser.parse_args()

    src = args.source_dir
    dst = args.destination_dir
    image_size = args.img_size

    start = time.perf_counter()

    transform_images_xmls(src, dst, image_size=image_size)
    print("\nTime Taken: ", round(time.perf_counter() - start, 3), "s")


if __name__ == "__main__":
    main()



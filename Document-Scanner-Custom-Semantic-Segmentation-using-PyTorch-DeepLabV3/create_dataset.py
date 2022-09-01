import os
import cv2
import time
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.utils import shuffle
from multiprocessing import Pool, cpu_count
from random import random, uniform, randint

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


def get_random_size(doc_height, doc_width, factor=None):
    size_factor = uniform(factor[0], factor[1])
    new_h, new_w = int(size_factor * doc_height), int(size_factor * doc_width)
    return new_h, new_w


def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width + 1
    max_y = image.shape[0] - crop_height + 1

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    ymin, xmin, ymax, xmax = y, x, y + crop_height, x + crop_width

    return ymin, xmin, ymax, xmax


def create(cropped_bck_img=None, doc_img=None, doc_msk=None):
    doc_img = doc_img / 255.0
    mask_inv = np.where(doc_msk == 255, 0.0, 1.0)

    cropped_bck_img_masked = cropped_bck_img * mask_inv
    merge_bck_and_true = ((cropped_bck_img_masked + doc_img) * 255).astype(np.int32)

    return merge_bck_and_true


def extract_image(image, startpoints, endpoints):
    transformed_img = F.perspective(image, startpoints, endpoints, fill=0, interpolation=T.InterpolationMode.NEAREST)

    x1, y1 = endpoints[0]
    x2, y2 = endpoints[1]
    x3, y3 = endpoints[2]
    x4, y4 = endpoints[3]

    ymin = min(y1, y2, y3, y4)
    xmin = min(x1, x2, x3, x4)

    height = abs(max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
    width = abs(min(x1, x2, x3, x4) - max(x1, x2, x3, x4))

    new = np.asarray(transformed_img)[ymin : ymin + height :, xmin : xmin + width, :]

    return new


def generate_perspective_transformed_image(transformer, distortion_scale, gen_count, image, mask, shape):

    W, H = shape
    random_idx_seed_value = randint(0, 1000000)

    torch.manual_seed(random_idx_seed_value)
    perspective_imgs = []

    for _ in range(gen_count):
        startpoints, endpoints = transformer.get_params(W, H, distortion_scale=distortion_scale)
        perspective_imgs.append(extract_image(image, startpoints, endpoints))

    torch.manual_seed(random_idx_seed_value)
    perspective_msks = []

    for _ in range(gen_count):
        startpoints, endpoints = transformer.get_params(W, H, distortion_scale=distortion_scale)
        perspective_msks.append(extract_image(mask, startpoints, endpoints))

    perspective_imgs, perspective_msks = shuffle(perspective_imgs, perspective_msks, random_state=1)

    return perspective_imgs, perspective_msks


def operation(params=None):

    prc_id = params["id"]
    print(f"[INFO] Starting process {prc_id}")

    DOC_IMGS = params["DOC_IMGS"]
    DOC_MSKS = params["DOC_MSKS"]
    BCK_IMGS = params["BCK_IMGS"]

    GEN_IMG_DIR = params["GEN_IMG_DIR"]
    GEN_MSK_DIR = params["GEN_MSK_DIR"]
    start_idx = params["start_idx"]

    motion_blur = A.MotionBlur(blur_limit=18, p=0.35)
    v_flip = A.VerticalFlip(p=0.75)
    h_flip = A.HorizontalFlip(p=0.75)
    rotate_aug = A.Rotate(border_mode=1, interpolation=0, p=0.5)
    color_J = A.ColorJitter(hue=0.25, saturation=0.25, p=0.7)

    # One of or apply more than one
    value = (0, 0, 0)
    if random() > 0.5:
        value = (255, 255, 255)

    opt_aug = A.OpticalDistortion(distort_limit=0.28, interpolation=0, border_mode=0, value=value, mask_value=0, p=0.9)
    grid_aug = A.GridDistortion(num_steps=2, distort_limit=(-0.22, 0.35), interpolation=0, border_mode=0, value=value, mask_value=0, p=0.9)
    elastic_aug = A.ElasticTransform(alpha=150, sigma=13, alpha_affine=10, interpolation=0, border_mode=0, value=value, mask_value=0, p=0.9)

    # One of
    compression_aug = A.ImageCompression(quality_lower=30, quality_upper=80, p=1.0)
    # downscale_aug = A.Downscale(p=1.0)
    noise = A.ISONoise(color_shift=(0.05, 0.25), p=0.75)

    # One of
    shadow = A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), num_shadows_lower=0, num_shadows_upper=1, shadow_dimension=3, p=0.7)
    sunflare = A.RandomSunFlare(
        flare_roi=(0.0, 0.0, 1.0, 1.0),
        angle_lower=0,
        angle_upper=1,
        num_flare_circles_lower=6,
        num_flare_circles_upper=10,
        src_radius=200,
        src_color=(255, 255, 255),
        p=0.6,
    )

    rgb_shift = A.RGBShift(r_shift_limit=20, g_shift_limit=0, b_shift_limit=10, p=0.4)
    cshuffle = A.ChannelShuffle(p=0.6)
    contrast = A.RandomBrightnessContrast(contrast_limit=(0.1, 0.34), p=0.5)
    contrast_2 = A.RandomBrightnessContrast(p=0.5)

    augs = A.Compose(
        [
            A.OneOf([v_flip, h_flip], p=0.8),
            rotate_aug,
            color_J,
            cshuffle,
            contrast_2,
            A.OneOf([opt_aug, grid_aug, elastic_aug], p=0.8),  # elastic_aug
            A.OneOf([noise, motion_blur, compression_aug], p=0.7),
            A.OneOf([shadow, sunflare, rgb_shift], p=0.65),  # rgb_shift
        ],
        p=1.0,
    )

    distortion_scale = 0.55
    perspective_transformer = T.RandomPerspective(distortion_scale=distortion_scale, p=0.7, interpolation=T.InterpolationMode.NEAREST)

    NUM_BCK_IMAGS = 6
    total_idxs = np.arange(0, len(BCK_IMGS))

    for doc_indx, (img_path, msk_path) in enumerate(zip(DOC_IMGS, DOC_MSKS), start_idx):
        orig_img = Image.open(img_path).convert("RGB")
        orig_msk = Image.open(msk_path).convert("RGB")

        W, H = orig_img.size

        # ========================================================
        perspective_imgs, perspective_msks = generate_perspective_transformed_image(
            transformer=perspective_transformer,
            distortion_scale=distortion_scale,
            gen_count=NUM_BCK_IMAGS,
            image=orig_img,
            mask=orig_msk,
            shape=(W, H),
        )

        random_bck_indx = np.random.choice(total_idxs, size=NUM_BCK_IMAGS, replace=False)
        bck_imgs_chosen = BCK_IMGS[random_bck_indx]

        for idx, bck_img_path in enumerate(bck_imgs_chosen):
            bck_img = cv2.imread(bck_img_path, cv2.IMREAD_COLOR)[:, :, ::-1]

            doc_img = perspective_imgs[idx]

            contrast_aug = contrast(image=doc_img)
            doc_img = contrast_aug["image"]

            doc_msk = perspective_msks[idx].astype(np.int32)

            height, width = doc_img.shape[0], doc_img.shape[1]

            # Random resize background image
            new_h, new_w = get_random_size(height, width, factor=(1.1, 1.4))
            bck_img = cv2.resize(bck_img, (new_w, new_h), cv2.INTER_CUBIC)

            # Random location in the background image
            ymin, xmin, ymax, xmax = get_random_crop(bck_img, height, width)
            cropped_bck_img = bck_img[ymin:ymax, xmin:xmax, :] / 255.0

            final_image = create(cropped_bck_img=cropped_bck_img, doc_img=doc_img, doc_msk=doc_msk)

            bck_img[ymin:ymax, xmin:xmax, :] = final_image

            # create a new mask
            new_mask = np.zeros_like(bck_img)
            new_mask[ymin:ymax, xmin:xmax, :] = doc_msk
            # print("Unique 1:", np.unique(new_mask))

            augmented = augs(image=bck_img, mask=new_mask)
            bck_img = augmented["image"]
            new_mask = augmented["mask"]

            bck_img = bck_img[:, :, ::-1]  # RGB to BGR for cv2.imwrite
            new_mask = new_mask.astype(np.uint8)

            assert len(np.unique(new_mask)) == 2

            new_save_name = f"{doc_indx:>04}_bck_{idx:>02}.png"

            cv2.imwrite(os.path.join(GEN_IMG_DIR, new_save_name), bck_img)
            cv2.imwrite(os.path.join(GEN_MSK_DIR, new_save_name), new_mask)

    print(f"[INFO] finishing process {prc_id}")
    return


def chunk(length, n):
    for i in range(0, length, n):
        yield (i, i + n)


if __name__ == "__main__":
    start_time = time.perf_counter()

    DOC_IMG_PATH = r"DOCUMENTS\\CHOSEN\\resized_images"
    DOC_MSK_PATH = r"DOCUMENTS\\CHOSEN\\resized_masks"

    GEN_IMG_DIR = r"final_set\\images"
    GEN_MSK_DIR = r"final_set\\masks"

    BCK_IMGS_DIR = r"background_images"

    DOC_IMGS = [os.path.join(DOC_IMG_PATH, i) for i in os.listdir(DOC_IMG_PATH)]
    DOC_MSKS = [os.path.join(DOC_MSK_PATH, i) for i in os.listdir(DOC_MSK_PATH)]

    BCK_IMGS = np.asarray([os.path.join(BCK_IMGS_DIR, i) for i in os.listdir(BCK_IMGS_DIR)])

    os.makedirs(GEN_IMG_DIR, exist_ok=True)
    os.makedirs(GEN_MSK_DIR, exist_ok=True)

    length = len(DOC_IMGS)
    procs = max(cpu_count() - 4, 2)
    print("Procs:", procs)

    numImagesPerProc = length / procs
    numImagesPerProc = int(np.ceil(numImagesPerProc))
    print("numImagesPerProc:", numImagesPerProc)

    CHUNKED_DOC_IMG_PATH = []
    CHUNKED_DOC_MSK_PATH = []
    startindxs = []

    for start, end in chunk(length, numImagesPerProc):
        CHUNKED_DOC_IMG_PATH.append(DOC_IMGS[start:end])
        CHUNKED_DOC_MSK_PATH.append(DOC_MSKS[start:end])
        startindxs.append(start)

    payloads = []

    # loop over the set chunked image paths
    for i, (doc_img_paths, doc_msk_paths, start_idx) in enumerate(zip(CHUNKED_DOC_IMG_PATH, CHUNKED_DOC_MSK_PATH, startindxs)):
        data = {
            "id": i,
            "DOC_IMGS": doc_img_paths,
            "DOC_MSKS": doc_msk_paths,
            "BCK_IMGS": BCK_IMGS,
            "GEN_IMG_DIR": GEN_IMG_DIR,
            "GEN_MSK_DIR": GEN_MSK_DIR,
            "start_idx": start_idx,
        }

        payloads.append(data)

    print("[INFO] Total images:", length)
    print(f"[INFO] launching pool using {procs} processes")

    pool = Pool(processes=procs)

    pool.map(operation, payloads)
    print("[INFO] waiting for processes to finish...")

    pool.close()
    pool.join()

    print("[INFO] multiprocessing complete")

    print("\nTime Taken: ", round(time.perf_counter() - start_time, 3), "s")

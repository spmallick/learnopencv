import os
import cv2
import shutil
import random
import numpy as np

np.random.seed(1)

og_img_dir = r"final_set\\images"
og_msk_dir = r"final_set\\masks"

# Saving resized images to reduce file size
MAX_DIM_SIZE = 480
img_per_doc = 6

train_img_dir = r"document_dataset_resized\\train\\images"
train_msk_dir = r"document_dataset_resized\\train\\masks"

valid_img_dir = r"document_dataset_resized\\valid\\images"
valid_msk_dir = r"document_dataset_resized\\valid\\masks"

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_msk_dir, exist_ok=True)
os.makedirs(valid_img_dir, exist_ok=True)
os.makedirs(valid_msk_dir, exist_ok=True)

all_img_paths = np.asarray(sorted([os.path.join(og_img_dir, i) for i in os.listdir(og_img_dir)]))
all_msk_paths = np.asarray(sorted([os.path.join(og_msk_dir, i) for i in os.listdir(og_msk_dir)]))

total_number_of_documents = len(all_img_paths) // img_per_doc


all_img_paths = np.split(all_img_paths, total_number_of_documents)
all_msk_paths = np.split(all_msk_paths, total_number_of_documents)


print(len(all_img_paths))


img_per_doc = list(range(img_per_doc))

train_img_paths = []
train_msk_paths = []
valid_img_paths = []
valid_msk_paths = []


for doc_id_img_paths, doc_id_msk_paths in zip(all_img_paths, all_msk_paths):

    number = random.choice(img_per_doc)

    for i in img_per_doc:
        if i == number:
            valid_img_paths.append(doc_id_img_paths[i])
            valid_msk_paths.append(doc_id_msk_paths[i])
        else:
            train_img_paths.append(doc_id_img_paths[i])
            train_msk_paths.append(doc_id_msk_paths[i])


print(len(train_img_paths), len(valid_img_paths))


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


def copy(img_paths, msk_paths, dst_img_dir, dst_msk_dir):

    for idx, (image_path, mask_path) in enumerate(zip(img_paths, msk_paths)):
        img_name = os.path.split(image_path)[-1]
        msk_name = os.path.split(mask_path)[-1]

        dst_image_path = os.path.join(dst_img_dir, img_name)
        dst_mask_path = os.path.join(dst_msk_dir, msk_name)

        # shutil.copyfile(image_path, dst_image_path)
        # shutil.copyfile(mask_path, dst_mask_path)

        # Saving resized images to reduce file size
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        asp_h, asp_w = ResizeWithAspectRatio(curr_dim=(h, w), resize_to=MAX_DIM_SIZE)

        image = cv2.resize(image, (asp_w, asp_h), interpolation=cv2.INTER_NEAREST)

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (asp_w, asp_h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(dst_image_path, image)
        cv2.imwrite(dst_mask_path, mask)

    return


# Training set
copy(train_img_paths, train_msk_paths, train_img_dir, train_msk_dir)

# Validation set
copy(valid_img_paths, valid_msk_paths, valid_img_dir, valid_msk_dir)


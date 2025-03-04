import torch
import cv2
import numpy as np
import os
import glob

from config import CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform


class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        """
        :param dir_path: Directory containing 'images/' and 'labels/' subfolders.
        :param width: Resized image width.
        :param height: Resized image height.
        :param classes: List of class names (or an indexing scheme).
        :param transforms: Albumentations transformations to apply.
        """
        self.transforms = transforms
        self.dir_path = dir_path
        self.image_dir = os.path.join(self.dir_path, "images")
        self.label_dir = os.path.join(self.dir_path, "labels")
        self.width = width
        self.height = height
        self.classes = classes

        # Gather all image paths
        self.image_file_types = ["*.jpg", "*.jpeg", "*.png", "*.ppm", "*.JPG"]
        self.all_image_paths = []
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.image_dir, file_type)))

        # Sort for consistent ordering
        self.all_image_paths = sorted(self.all_image_paths)
        self.all_image_names = [os.path.basename(img_p) for img_p in self.all_image_paths]

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        # 1) Read image
        image_name = self.all_image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_filename = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # 2) Resize image (to the model's expected size)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0  # Scale pixel values to [0, 1]

        # 3) Read bounding boxes (normalized) from .txt file
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Format: class_id x_min y_min x_max y_max  (all in [0..1])
                parts = line.split()
                class_id = int(parts[0])  # e.g. 0, 1, 2, ...
                xmin = float(parts[1])
                ymin = float(parts[2])
                xmax = float(parts[3])
                ymax = float(parts[4])

                # Example: if you want class IDs to start at 1 for foreground
                # and background=0, do:
                label_idx = class_id + 1

                # Convert normalized coords to absolute (in resized space)
                x_min_final = xmin * self.width
                y_min_final = ymin * self.height
                x_max_final = xmax * self.width
                y_max_final = ymax * self.height

                # Ensure valid box
                if x_max_final <= x_min_final:
                    x_max_final = x_min_final + 1
                if y_max_final <= y_min_final:
                    y_max_final = y_min_final + 1

                # Clip if out of bounds
                x_min_final = max(0, min(x_min_final, self.width - 1))
                x_max_final = max(0, min(x_max_final, self.width))
                y_min_final = max(0, min(y_min_final, self.height - 1))
                y_max_final = max(0, min(y_max_final, self.height))

                boxes.append([x_min_final, y_min_final, x_max_final, y_max_final])
                labels.append(label_idx)

        # 4) Convert boxes & labels to Torch tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # 5) Prepare the target dict
        area = (
            (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if len(boxes) > 0
            else torch.tensor([], dtype=torch.float32)
        )
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd, "image_id": image_id}

        # 6) Albumentations transforms: pass Python lists, not Tensors
        if self.transforms:
            bboxes_list = boxes.cpu().numpy().tolist()  # shape: list of [xmin, ymin, xmax, ymax]
            labels_list = labels.cpu().numpy().tolist()  # shape: list of ints

            transformed = self.transforms(
                image=image_resized,
                bboxes=bboxes_list,
                labels=labels_list,
            )

            # Reassign the image
            image_resized = transformed["image"]

            # Convert bboxes back to Torch Tensors
            new_bboxes_list = transformed["bboxes"]  # list of [xmin, ymin, xmax, ymax]
            new_labels_list = transformed["labels"]  # list of int

            if len(new_bboxes_list) > 0:
                new_bboxes = torch.tensor(new_bboxes_list, dtype=torch.float32)
                new_labels = torch.tensor(new_labels_list, dtype=torch.int64)
            else:
                new_bboxes = torch.zeros((0, 4), dtype=torch.float32)
                new_labels = torch.zeros((0,), dtype=torch.int64)

            target["boxes"] = new_bboxes
            target["labels"] = new_labels

        return image_resized, target


# ---------------------------------------------------------
# Create train/valid datasets and loaders
# ---------------------------------------------------------
def create_train_dataset(DIR):
    train_dataset = CustomDataset(
        dir_path=DIR, width=RESIZE_TO, height=RESIZE_TO, classes=CLASSES, transforms=get_train_transform()
    )
    return train_dataset


def create_valid_dataset(DIR):
    valid_dataset = CustomDataset(
        dir_path=DIR, width=RESIZE_TO, height=RESIZE_TO, classes=CLASSES, transforms=get_valid_transform()
    )
    return valid_dataset


def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return valid_loader


# ---------------------------------------------------------
# Debug/demo if run directly
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example usage with no transforms for debugging
    dataset = CustomDataset(dir_path=TRAIN_DIR, width=RESIZE_TO, height=RESIZE_TO, classes=CLASSES, transforms=None)
    print(f"Number of training images: {len(dataset)}")

    def visualize_sample(image, target):
        """
        Visualize a single sample using OpenCV. Expects
        `image` as a NumPy array of shape (H, W, 3) in [0..1].
        """
        # Convert [0,1] float -> [0,255] uint8
        img = (image * 255).astype(np.uint8)
        # Convert RGB -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        boxes = target["boxes"].cpu().numpy().astype(np.int32)
        labels = target["labels"].cpu().numpy().astype(np.int32)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_idx = labels[i]

            # If your class_idx starts at 1 for "first class", ensure you handle that:
            # e.g. if CLASSES = ["background", "class1", "class2", ...]
            if 0 <= class_idx < len(CLASSES):
                class_str = CLASSES[class_idx]
            else:
                class_str = f"Label_{class_idx}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, class_str, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Sample", img)
        cv2.waitKey(0)

    # Visualize a few samples
    NUM_SAMPLES_TO_VISUALIZE = 10
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]  # No transforms in this example
        # `image` is shape (H, W, 3) in [0..1]
        print(f"Visualizing sample {i}, boxes: {target['boxes'].shape[0]}")
        visualize_sample(image, target)
    cv2.destroyAllWindows()

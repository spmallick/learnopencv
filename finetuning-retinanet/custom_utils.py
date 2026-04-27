import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES, BATCH_SIZE

plt.style.use("ggplot")


class Averager:
    """
    A class to keep track of running average of values (e.g. training loss).
    """

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:
    """
    Saves the model if the current epoch's validation mAP is higher
    than all previously observed values.
    """

    def __init__(self, best_valid_map=float(0)):
        self.best_valid_map = best_valid_map

    def __call__(
        self,
        model,
        current_valid_map,
        epoch,
        OUT_DIR,
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"SAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                },
                f"{OUT_DIR}/best_model.pth",
            )


def collate_fn(batch):
    """
    To handle the data loading as different images may have different
    numbers of objects, and to handle varying-size tensors as well.
    """
    return tuple(zip(*batch))


def get_train_transform():
    # We keep "pascal_voc" because bounding box format is [x_min, y_min, x_max, y_max].
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45),
            A.Blur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
            A.RandomScale(scale_limit=0.2, p=0.3),
            ToTensorV2(p=1.0),
        ],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def get_valid_transform():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def show_tranformed_image(train_loader):
    """
    Visualize transformed images from the `train_loader` for debugging.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(2):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)

            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            for i in range(len(images)):
                if len(targets[i]["boxes"]) == 0:
                    continue
                boxes = targets[i]["boxes"].cpu().numpy().astype(np.int32)
                labels = targets[i]["labels"].cpu().numpy().astype(np.int32)
                sample = images[i].permute(1, 2, 0).cpu().numpy()
                sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                sample = (sample * 255).astype(np.uint8)
                for box_num, box in enumerate(boxes):
                    cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(
                        sample,
                        CLASSES[labels[box_num]],
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )
                cv2.imshow("Transformed image", sample)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def save_model(epoch, model, optimizer):
    """
    Save the trained model (state dict) and optimizer state to disk.
    """
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "outputs/last_model.pth",
    )


def save_loss_plot(OUT_DIR, train_loss_list, x_label="iterations", y_label="train loss", save_name="train_loss"):
    """
    Saves the training loss curve.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_list, color="tab:blue")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"{OUT_DIR}/{save_name}.png")
    # plt.close()
    print("SAVING PLOTS COMPLETE...")


def save_mAP(OUT_DIR, map_05, map):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 curves per epoch.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(map_05, color="tab:orange", linestyle="-", label="mAP@0.5")
    plt.plot(map, color="tab:red", linestyle="-", label="mAP@0.5:0.95")
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/map.png")
    # plt.close()
    print("SAVING mAP PLOTS COMPLETE...")

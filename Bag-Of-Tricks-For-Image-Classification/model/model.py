import os
import warnings
from argparse import (
    ArgumentParser,
    Namespace,
)

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .augmentations import (
    get_test_augmentation,
    get_training_augmentation,
)
from .losses import (
    KnowledgeDistillationLoss,
    LabelSmoothingLoss,
    MixUpAugmentationLoss,
)


class LitFood101(pl.LightningModule):
    def __init__(self, model, args: Namespace):
        super().__init__()
        self.model = model
        self.args = args
        # We need to specify a number of classes there to avoid the RuntimeError
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/3006
        # However, we will get another warning and it should be handled in forward steps
        self.metric = pl.metrics.Accuracy(num_classes=self.args.num_classes)
        dim_feats = self.model.fc.in_features  # =2048
        nb_classes = self.args.num_classes
        self.model.fc = nn.Linear(dim_feats, nb_classes)

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        if self.args.use_smoothing:
            self.criterion = LabelSmoothingLoss(
                self.args.num_classes, self.args.smoothing,
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        if self.args.use_mixup:
            self.criterion = MixUpAugmentationLoss(self.criterion)

    def on_epoch_start(self):
        self.previous_batch = [None, None]

    def training_step(self, batch, *args):
        x, y = batch[0]["image"], batch[1]
        if self.args.use_mixup:
            mixup_x, *mixup_y = self.mixup_batch(x, y, *self.previous_batch)
            logits = self(mixup_x)
            loss = self.criterion(logits, mixup_y)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
        # We ignore a warning about a mismatch between a number of predicted classes
        # and a number of initialized for Accuracy class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accuracy = self.metric(logits.argmax(dim=-1), y)
        tensorboard_logs = {"train_loss": loss, "train_acc": accuracy}
        self.previous_batch = [x, y]

        return {"loss": loss, "progress_bar": tensorboard_logs, "log": tensorboard_logs}

    def validation_step(self, batch, *args):
        x, y = batch[0]["image"], batch[1]
        logits = self(x)
        val_loss = self.criterion(logits, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_accuracy = self.metric(logits.argmax(dim=-1), y)
        return {"val_loss": val_loss, "val_acc": val_accuracy}

    def test_step(self, batch, *args):
        x, y = batch[0]["image"], batch[1]
        logits = self(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_accuracy = self.metric(logits.argmax(dim=-1), y)
        return {"test_acc": test_accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["val_acc"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_accuracy}
        return {
            "avg_val_loss": avg_loss,
            "avg_val_acc": avg_accuracy,
            "log": tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        avg_accuracy = torch.stack([x["test_acc"] for x in outputs]).mean()
        return {"avg_test_acc": avg_accuracy.item()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.use_cosine_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.max_epochs, eta_min=0.0,
            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.args.milestones,
            )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ImageFolder(
            os.path.join(self.args.data_root, "train"),
            transform=get_training_augmentation(),
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataset = ImageFolder(
            os.path.join(self.args.data_root, "test"),
            transform=get_test_augmentation(),
        )
        return DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def optimizer_step(self, epoch, batch_idx, optimizer, *args, **kwargs):
        # Learning Rate warm-up
        if self.args.warmup != -1 and epoch < self.args.warmup:
            lr = self.args.lr * (epoch + 1) / self.args.warmup
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        self.logger.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, step=epoch)
        optimizer.step()
        optimizer.zero_grad()

    def mixup_batch(self, x, y, x_previous, y_previous):
        lmbd = (
            np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
            if self.args.mixup_alpha > 0
            else 1
        )
        if x_previous is None:
            x_previous = torch.empty_like(x).copy_(x)
            y_previous = torch.empty_like(y).copy_(y)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        # If current batch size != previous batch size, we take only a part of the previous batch
        x_previous = x_previous[:batch_size, ...]
        y_previous = y_previous[:batch_size, ...]
        x_mixed = lmbd * x + (1 - lmbd) * x_previous[index, ...]
        y_a, y_b = y, y_previous[index]
        return x_mixed, y_a, y_b, lmbd

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data-root",
            default="./data",
            type=str,
            help="Path to root folder of the dataset (should include train and test folders)",
        )
        parser.add_argument(
            "-n", "--num-classes", type=int, help="Number of classes", default=21,
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            default=32,
            type=int,
            metavar="N",
            help="Mini-batch size",
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=1e-4,
            type=float,
            metavar="LR",
            help="Initial learning rate",
        )
        parser.add_argument(
            "--milestones",
            type=int,
            nargs="+",
            default=[15, 30],
            help="Milestones for dropping the learning rate",
        )

        parser.add_argument(
            "--warmup",
            type=int,
            default=6,
            help="Number of epochs to warm up the learning rate. -1 to turn off",
        )
        return parser


class LitFood101KD(LitFood101):
    def __init__(self, model, teacher, args):
        super().__init__(model, args)
        self.teacher = teacher
        dim_feats = self.teacher.fc.in_features  # =2048
        nb_classes = self.args.num_classes
        self.teacher.fc = nn.Linear(dim_feats, nb_classes)
        teacher_checkpoint = torch.load("./teacher.ckpt")
        self.teacher.load_state_dict(teacher_checkpoint["state_dict"])

    def setup(self, stage):
        criterion = (
            LabelSmoothingLoss(self.args.num_classes, self.args.smoothing)
            if self.args.use_smoothing
            else nn.CrossEntropyLoss()
        )
        self.criterion = KnowledgeDistillationLoss(
            self.args.distill_alpha, self.args.distill_temperature, criterion=criterion,
        )
        if self.args.use_mixup:
            self.criterion = MixUpAugmentationLoss(self.criterion)
        self.teacher.eval()

    def training_step(self, batch, *args):
        x, y = batch[0]["image"], batch[1]
        with torch.no_grad():
            teacher_output = self.teacher(x)

        if self.args.use_mixup:
            mixup_x, *mixup_y = self.mixup_batch(x, y, *self.previous_batch)
            logits = self(mixup_x)
            loss = self.criterion(logits, mixup_y, teacher_output)
        else:
            logits = self(x)
            loss = self.criterion(logits, y, teacher_output)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accuracy = self.metric(logits.argmax(dim=-1), y)
        tensorboard_logs = {"train_loss": loss, "train_acc": accuracy}

        return {"loss": loss, "progress_bar": tensorboard_logs, "log": tensorboard_logs}

    def validation_step(self, batch, *args):
        x, y = batch[0]["image"], batch[1]
        logits = self(x)
        with torch.no_grad():
            teacher_output = self.teacher(x)
        val_loss = self.criterion(logits, y, teacher_output)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_accuracy = self.metric(logits.argmax(dim=-1), y)
        return {"val_loss": val_loss, "val_acc": val_accuracy}

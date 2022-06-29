#Author: Drishttii@

#!pip install pytorch-lightning

#!pip install pytorch-lightning-bolts

import superimport

import torch
import torchvision
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import EMNIST
import torchvision.transforms as tt
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from typing import Any, Callable, Optional, Sequence, Union
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

# %matplotlib inline
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

class EMNISTDataModule(VisionDataModule):

    name = "emnist"
    dataset_cls = EMNIST
    dims = (1, 28, 28)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 4,
        normalize: bool = False,
        batch_size: int = 400,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_transforms = tt.Compose([
                      lambda img: tt.functional.rotate(img, -90),
                      lambda img: tt.functional.hflip(img),
                      tt.ToTensor()
                  ]),
            val_transforms = tt.Compose([
                      lambda img: tt.functional.rotate(img, -90),
                      lambda img: tt.functional.hflip(img),
                      tt.ToTensor()
                  ]),
            test_transforms = tt.Compose([
                      lambda img: tt.functional.rotate(img, -90),
                      lambda img: tt.functional.hflip(img),
                      tt.ToTensor()
                  ]),
            *args,
            **kwargs,
        )

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir
        """
        self.dataset_cls(self.data_dir, split="byclass", train=True, download=True, transform=tt.Compose([
                      lambda img: tt.functional.rotate(img, -90),
                      lambda img: tt.functional.hflip(img),
                      tt.ToTensor()
                  ]))
        self.dataset_cls(self.data_dir, split="byclass", train=False, download=True, transform=tt.Compose([
                      lambda img: tt.functional.rotate(img, -90),
                      lambda img: tt.functional.hflip(img),
                      tt.ToTensor()
                  ]))

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset
        """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(self.data_dir, split="byclass", train=True, transform=train_transforms, **self.EXTRA_ARGS)
            dataset_val = self.dataset_cls(self.data_dir, split="byclass", train=True, transform=val_transforms, **self.EXTRA_ARGS)

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, split="byclass", train=False, transform=test_transforms, **self.EXTRA_ARGS
            )

    def show_batch(self, dl):
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=20).permute(1, 2, 0))
            break

    # This helper funcation convert the output index [0-61] into character [0-9],[A-Z],[a-z]
    def to_char(self, num):
        if num<10:
            return str(num)
        elif num < 36:
            return chr(num+55)
        else:
            return chr(num+61)

    # This is reverse of above function. Convert character [0-9],[A-Z],[a-z] into index [0-61]
    def to_index(self, char):
        if ord(char)<59:
            return ord(char)-48
        elif ord(char)<95:
            return ord(char)-55
        else:
            return ord(char)-61

    def show_example(self, data):
        img, label = data
        print("Label: ("+self.to_char(label)+")")
        plt.imshow(img[0], cmap="gray")

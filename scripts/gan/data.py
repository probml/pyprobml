from functools import partial
import pandas as pd
import os
import PIL
import glob

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, io
from torchvision.datasets.utils import verify_str_arg

import pytorch_lightning as pl


class CelebADataset(Dataset):
    """CelebA Dataset class"""

    def __init__(self, 
                 root,
                 split="train",
                 target_type="attr",
                 transform=None,
                 target_transform=None,
                 download=False
                 ):
        """
        """

        self.root = root
        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download_from_kaggle()

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pd.read_csv(fn("list_eval_partition.csv"), delim_whitespace=False, header=0, index_col=0)
        # This file is not available in Kaggle
        # identity = pd.read_csv(fn("identity_CelebA.csv"), delim_whitespace=True, header=None, index_col=0)
        bbox = pd.read_csv(fn("list_bbox_celeba.csv"), delim_whitespace=False, header=0, index_col=0)
        landmarks_align = pd.read_csv(fn("list_landmarks_align_celeba.csv"), delim_whitespace=False, header=0, index_col=0)
        attr = pd.read_csv(fn("list_attr_celeba.csv"), delim_whitespace=False, header=0, index_col=0)

        mask = slice(None) if split_ is None else (splits['partition'] == split_)

        self.filename = splits[mask].index.values
        # self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def download_from_kaggle(self):

        # Annotation files will be downloaded at the end
        label_files = ['list_attr_celeba.csv', 'list_bbox_celeba.csv', 'list_eval_partition.csv', 'list_landmarks_align_celeba.csv']

        # Check if files have been downloaded already
        files_exist = False
        for label_file in label_files:
            if os.path.isfile(os.path.join(self.root, label_file)):
                files_exist = True
            else:
                files_exist = False

        if files_exist:
            print("Files exist already")
        else:
            print("Downloading dataset. Please while while the download and extraction processes complete")
            # Download files from Kaggle using its API as per
            # https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python

            # Kaggle authentication
            # Remember to place the API token from Kaggle in $HOME/.kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            # Download all files of a dataset
            # Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
            api.dataset_download_files(dataset='jessicali9530/celeba-dataset',
                                       path=self.root,
                                       unzip=True,
                                       force=False,
                                       quiet=False)

            # Downoad the label files
            # Signature: dataset_download_file(dataset, file_name, path=None, force=False, quiet=True)
            for label_file in label_files:
                api.dataset_download_file(dataset='jessicali9530/celeba-dataset',
                                          file_name=label_file,
                                          path=self.root,
                                          force=False,
                                          quiet=False)

            # Clear any remaining *.csv.zip files
            files_to_delete = glob.glob(os.path.join(self.root,"*.csv.zip"))
            for f in files_to_delete:
                os.remove(f)

            print("Done!")


    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root, 
                                        "img_align_celeba", 
                                        "img_align_celeba", 
                                        self.filename[index]))

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            # elif t == "identity":
            #     target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(f"Target type {t} is not recognized")

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)


class CelebADataModule(pl.LightningDataModule):

    def __init__(self, 
                 data_dir,
                 target_type="attr",
                 train_transform=None,
                 val_transform=None,
                 target_transform=None,
                 download=False,
                 batch_size=32,
                 num_workers=8):

        super().__init__()

        self.data_dir = data_dir
        self.target_type = target_type
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.target_transform = target_transform
        self.download = download

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Training dataset
        self.celebA_trainset = CelebADataset(root=self.data_dir,
                                             split='train',
                                             target_type=self.target_type,
                                             download=self.download,
                                             transform=self.train_transform,
                                             target_transform=self.target_transform)

        # Validation dataset
        self.celebA_valset = CelebADataset(root=self.data_dir,
                                           split='valid',
                                           target_type=self.target_type,
                                           download=False,
                                           transform=self.val_transform,
                                           target_transform=self.target_transform)

        # Test dataset
        self.celebA_testset = CelebADataset(root=self.data_dir,
                                            split='test',
                                            target_type=self.target_type,
                                            download=False,
                                            transform=self.val_transform,
                                            target_transform=self.target_transform)

    def train_dataloader(self):
        return DataLoader(self.celebA_trainset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.celebA_valset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.celebA_testset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

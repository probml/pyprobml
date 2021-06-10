"""
This file has functions which return data loaders for CLIP extracted features for
different datasets.
Author: Srikar-Reddy-Jilugu(@always-newbie161)
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from zipfile import ZipFile
import random
import numpy as np
import os
import wget

torch.manual_seed(0)
random.seed(0)


class _Clip_ds(Dataset):

    def __init__(self, images, labels):
        self.X = images
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        label = self.labels[index]
        sample = (image, label)
        return sample

    transform = transforms.Compose([
    ])


def get_imagenette_clip_loaders(dir_name='', train_shuffle=False):

    fname_train = os.path.join(dir_name, 'imagenette_clip_data.pt.zip')

    if not os.path.exists(fname_train):
        print("train_data zip-file not available in this dir, downloading from source...")
        url = 'https://github.com/probml/probml-data/blob/main/data/imagenette_clip_data.pt.zip?raw=true'
        wget.download(url, fname_train)
        print("train_data zip-file downloaded successfully")

    zip_train = ZipFile(fname_train, 'r')

    fname_test = os.path.join(dir_name, 'imagenette_test_clip_data.pt.zip')

    if not os.path.exists(fname_test):
        print("test_data zip-file not available in this dir, downloading from source...")
        url = 'https://github.com/probml/probml-data/blob/main/data/imagenette_test_clip_data.pt.zip?raw=true'
        wget.download(url, fname_test)
        print("test_data zip-file downloaded successfully")

    zip_test = ZipFile(fname_test, 'r')

    zip_train.extractall()
    zip_test.extractall()
    train_data = torch.load('imagenette_clip_data.pt')
    test_data = torch.load('imagenette_test_clip_data.pt')

    train_features = train_data[:, :-1].to(torch.float32)
    test_features = test_data[:, :-1].to(torch.float32)
    train_labels = train_data[:, -1].numpy().astype(np.int64)
    test_labels = test_data[:, -1].numpy().astype(np.int64)

    print(f'size of training_data: features:{train_features.shape}, labels:{train_labels.shape}')
    print(f'size of testing_data: features:{test_features.shape}, labels:{test_labels.shape}')

    train_dataset = _Clip_ds(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=train_shuffle, num_workers=4)

    test_dataset = _Clip_ds(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    return train_loader, test_loader


def get_test_data(dir_name=''):

    _, test_loader = get_imagenette_clip_loaders(dir_name=dir_name, train_shuffle=False)
    test_features, test_labels = [], []
    for features, labels in test_loader:
        test_features.append(features)
        test_labels.append(labels)

    test_features, test_labels = torch.cat(test_features), torch.cat(test_labels)

    return test_features, test_labels

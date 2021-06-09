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


def get_imagenette_clip_loaders(train_shuffle=False):
    zip_train = ZipFile('../data/imagenette_clip_data.pt.zip', 'r')
    zip_test = ZipFile('../data/imagenette_test_clip_data.pt.zip', 'r')
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

def test():
    #from clip_dataloaders import get_imagenette_clip_loaders
    train_loader, test_loader = get_imagenette_clip_loaders(train_shuffle=False)
    test_features, test_labels = [], []
    for features, labels in test_loader:
        test_features.append(features)
        test_labels.append(labels)
    return test_features, test_labels



# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml


import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

# We need to rotate the images
# https://github.com/pytorch/vision/issues/2630

transform=transforms.Compose([lambda img: torchvision.transforms.functional.rotate(img, -90),
                                transforms.RandomHorizontalFlip(p=1),
                                transforms.ToTensor()])

training_data = datasets.EMNIST(
    root="~/data",
    split="byclass",
    download=True,
    transform=transform
)

figure = plt.figure(figsize=(10, 10))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    #sample_idx = torch.randint(len(training_data), size=(1,)).item()
    sample_idx = i
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    label = training_data.classes[label]
    plt.title(label, fontsize=18)
    #plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)

plt.tight_layout()
pml.savefig("emnist-data.pdf")
plt.show()

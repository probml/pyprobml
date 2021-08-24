# Based on
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb
# (MIT License)

import superimport

from __future__ import absolute_import, division, print_function


from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
figdir = "../figures"


def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


# print(tf.__version__)
np.random.seed(0)


data = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# print(np.shape(train_images))
# print(np.shape(test_images))


# For CIFAR:
# (50000, 32, 32, 3)
# (10000, 32, 32, 3)

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    y = train_labels[i][0]
    plt.xlabel(class_names[y])
save_fig("cifar10-data.pdf")
plt.show()

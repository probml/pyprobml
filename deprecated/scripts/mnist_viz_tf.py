


import superimport

from tensorflow import keras
import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from time import time
import os
figdir = "../figures"


def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


# print(tf.__version__)
np.random.seed(0)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# print(np.shape(train_images))
# print(np.shape(test_images))
#(60000, 28, 28)
#(10000, 28, 28)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
save_fig("mnist-data.pdf")
plt.show()
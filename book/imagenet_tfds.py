#https://www.tensorflow.org/datasets/datasets#imagenet2012

import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds

#tf.enable_eager_execution()

# See all registered datasets
tfds.list_builders()

# Load a given dataset by name, along with the DatasetInfo
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']
assert isinstance(train_data, tf.data.Dataset)
assert info.features['label'].num_classes == 10
assert info.splits['train'].num_examples == 60000

# You can also access a builder directly
builder = tfds.builder("mnist")
assert builder.info.splits['train'].num_examples == 60000
builder.download_and_prepare()
datasets = builder.as_dataset()

# If you need NumPy arrays
np_datasets = tfds.as_numpy(datasets)

#data, info = tfds.load("Imagenet2012", with_info=True)
# tfds.image.imagenet.Imagenet2012
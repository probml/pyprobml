#https://www.tensorflow.org/datasets/datasets#imagenet2012

import superimport

import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds


np.random.seed(42)

#tf.enable_eager_execution() # enabled by default for 2.0

builder = tfds.builder("mnist")
builder.download_and_prepare()
info = builder.info
assert info.features['label'].num_classes == 10
assert info.splits['train'].num_examples == 60000
print(info.features['label'].names) 
## ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Create numpy generator of numpy arrays.
# as_supervised=True gives us the (image, label) as a tuple instead of a dict
datastream = builder.as_dataset(as_supervised=True, split='train')
datastream_small = tfds.as_numpy(datastream.batch(3).take(2))
#datastream_small = datastream.batch(3).take(2)
for image_batch, label_batch in datastream_small:
    print(image_batch.shape) 
    print(label_batch)
## (3, 28, 28, 1)
## [9 0 3]
## (3, 28, 28, 1)
## [7 7 8]

# We can also iterate over the data indefinitely
datastream = builder.as_dataset(as_supervised=True, split='train', shuffle_files=False)
datastream_inf = tfds.as_numpy(datastream.batch(3).take(2).repeat())
i = 0
for image_batch, label_batch in datastream_inf:
    print(image_batch.shape) 
    print(label_batch)
    i += 1
    if i > 4: break

'''
# Fetch full dataset and convert from TF tensors to numpy arrays.
mnist_data, info = tfds.load(name='mnist', batch_size=-1, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']
num_labels = info.features['label'].num_classes
h, w, c = info.features['image'].shape
num_pixels = h * w * c
train_images, train_labels = train_data['image'], train_data['label']
print(train_images.shape) ## (60000, 28, 28, 1)

 
# Create datastream.
# as_supervised=True gives us the (image, label) as a tuple instead of a dict
ds = tfds.load(name='mnist', split='train', as_supervised=True)
bs = 3
train_small = tfds.as_numpy(ds.batch(bs).take(2)) # Numpy generator
for image_batch, label_batch in train_small:
    print(image_batch.shape) ## (3, 28, 28, 1)
    image_batch_flat = np.reshape(image_batch[:, :, :, 0], (bs, num_pixels))
    print(image_batch_flat.shape) ## (3, 784)
    print(label_batch)


# As usual with TF, there are many ways to do the same thing :(
# Method 1
datasets, info = tfds.load("mnist", with_info=True)
#data2, info2 = tfds.load("imagenet2012", with_info=True)
train_data, test_data = datasets['train'], datasets['test']
#mnist_train = tfds.load(name="mnist", split=tfds.Split.TRAIN)

# Method 2
builder = tfds.builder("mnist")
info = builder.info
assert info.features['label'].num_classes == 10
assert info.splits['train'].num_examples == 60000
print(info.features['label'].names) 
## ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(info)

    
builder.download_and_prepare()
datasets = builder.as_dataset() ## dict of two TF datasets
train_data, test_data = datasets['train'], datasets['test']
#train_data = builder.as_dataset(split=tfds.Split.TRAIN)
assert isinstance(train_data, tf.data.Dataset)

# A TF dataset is a TF generator. We can extract elements as follows.
example, = train_data.take(1)

# Each example is a dictionary of tensors
image, label = example["image"], example["label"]
print(type(image))  ## tensorflow.python.framework.ops.EagerTensor
print(image.shape) ## (28, 28, 1)

# We can make each example be a tuple of tensors instead of a dict
datasets = builder.as_dataset(as_supervised=True) ## dict of two datasets
train_data, test_data = datasets['train'], datasets['test']
example, = train_data.take(1)
image = example[0], label = example[1]

# We can convert tensors to  numpy as follows:
img = image.numpy()[:, :, 0].astype(np.float32)
print(type(img)) ## np.ndarray

#  We can also convert the whole dataset to a numpy generator
datasets = builder.as_dataset(as_supervised=True) ## dict of two datasets
np_datasets = tfds.as_numpy(datasets)
np_train_data, np_test_data = np_datasets['train'], np_datasets['test']
image, label = next(np_train_data)
assert isinstance(image, np.ndarray)


# We can create batches of numpy data thus:
data = tfds.as_numpy(train_data.batch(3).take(2))
for image_batch, label_batch in data:
    print(image_batch.shape) ## (3, 28, 28, 1)
    print(label_batch)
    
'''




# Illustrate how to read IMDB data
# https://machinetalk.org/2019/03/18/introduction-to-tensorflow-datasets/

import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds


np.random.seed(42)

imdb, info = tfds.load('imdb_reviews/subwords32k', with_info=True, as_supervised=True)
train_dataset, test_data = imdb['train'], imdb['test']
train_data = train_dataset.batch(1).take(1)

for data, label in train_data:
    print(data)

tokenizer = info.features['text'].encoder
for data, label in train_data:
    print(tokenizer.decode(data.numpy()[0]))
    
bs = 3 # batch size
train_data = train_dataset.padded_batch(bs, train_dataset.output_shapes)
for i, (data, label) in enumerate(train_data):
    print(data.shape) 
    if i > 4: break
'''
(3, 228)
(3, 248)
(3, 298)
(3, 210)
(3, 727)
'''

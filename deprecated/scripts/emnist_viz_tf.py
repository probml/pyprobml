


import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

import tensorflow as tf
import tensorflow_datasets as tfds

np.random.seed(0)

ds, info = tfds.load('emnist', split='test', shuffle_files=False, with_info=True) # horribly slow
print(info)


plt.figure(figsize=(10, 10))
i = 0
for example in ds:
    image = example["image"]
    label = example["label"]
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image)
    plt.title(label)
    i += 1
    if i >= 25: break

pml.savefig("emnist-data.pdf")
plt.show()
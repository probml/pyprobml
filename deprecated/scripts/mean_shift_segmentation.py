# Mean shift for segmenting an image by clustering pixels based on color and proximity
#https://github.com/log0/build-your-own-meanshift/blob/master/Meanshift%20Image%20Segmentation.ipynb
#http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/

import superimport

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
import io
import pyprobml_utils as pml
import os

#%matplotlib inline
#pylab.rcParams['figure.figsize'] = 16, 12

r = requests.get('https://github.com/probml/probml-data/blob/main/data/bread.jpg?raw=true', stream=True)
image = Image.open(io.BytesIO(r.content))

# Image is (687 x 1025, RGB channels)
image = np.array(image)
original_shape = image.shape

# Flatten image.
X = np.reshape(image, [-1, 3])

plt.figure()
plt.imshow(image)
plt.axis('off')
pml.savefig('meanshift_segmentation_input.pdf')

bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
print("bandwidth {}".format(bandwidth))

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

segmented_image = np.reshape(labels, original_shape[:2])  # Just take (height, width), ignore color dim.

plt.figure()
plt.imshow(segmented_image)
plt.axis('off')
pml.savefig('meanshift_segmentation_result.pdf')

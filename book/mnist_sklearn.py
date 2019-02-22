# Poking around the MNIST dataset.
# Code based on 
#https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from utils import save_fig    
  
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.keys()
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'details', 'categories', 'url'])

X, y = mnist["data"], mnist["target"]
X = X / 255 # convert to real in [0,1] 
y = y.astype(np.uint8)
print(X.shape) # (70000, 784)
# Standard train/ test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    montage = np.concatenate(row_images, axis=0)
    plt.imshow(montage, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    
plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
save_fig("mnist_digits")
plt.show()


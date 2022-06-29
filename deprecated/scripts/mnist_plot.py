# Visualize MNIST dataset
# Code based on 
# https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb



import superimport

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
figdir ="../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml


def load_mnist_data_openml():
  # Returns X_train: (60000, 784), X_test: (10000, 784), scaled [0...1]
  # y_train: (60000,) 0..9 ints, y_test: (10000,)
    print("Downloading mnist...")
    data = fetch_openml('mnist_784', version=1, cache=True)
    print("Done")
    #data = fetch_mldata('MNIST original')
    X = data['data'].astype('float32')
    y = data["target"].astype('int64')
    # Normalize features
    X = X / 255
    # Create train-test split (as [Joachims, 2006])
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train, X_test, y_train, y_test

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


X_train, X_test, y_train, y_test = load_mnist_data_openml()
 
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
example_images = X_train[:100]
plot_digits(example_images, images_per_row=10)
save_fig("mnist_digits")
plt.show()
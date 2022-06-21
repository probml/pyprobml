# PCA in 2d on digit images. 
# Based on fig 14.23 of  of "Elements of statistical learning". Code is from Andrey Gaskov's site:

# Code modified from    
# https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/examples/ZIP%20Code.ipynb

import superimport

import pandas as pd
from matplotlib import transforms, pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import requests
from io import BytesIO
import pyprobml_utils as pml
from tensorflow import keras
import tensorflow as tf

# define plots common properties and color constants
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
GRAY1, GRAY4, GRAY7 = '#231F20', '#646369', '#929497'

if 1:
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    ndx = np.arange(1000)
    nsamples, nrows, ncols = train_images.shape
    X_train = np.reshape(train_images, (nsamples, nrows*ncols))
    X_train = X_train[ndx,:]
    y_train = train_labels[ndx]
    w = 30 # 28+2

if 0:
    digits = datasets.load_digits()
    X_train = digits.data
    y_train = digits.target # 0..9
    w = 10  # pixels for one digit, 8+2


if 0:
    # load numpy array from the compressed file 
    url = 'https://github.com/probml/probml-data/blob/main/data/goog.npy?raw=true'
    response = requests.get(url)
    rawdata = BytesIO(response.content)
    arr = np.load(rawdata)['arr_0'] #'../data/zip.npy.npz'
    # do train-test split by the last column
    train, test = arr[arr[:, -1] == 0], arr[arr[:, -1] == 1]
    X_train, X_test = train[:, 1:-1], test[:, 1:-1]
    y_train, y_test = train[:, 0].astype(int), test[:, 0].astype(int)
    w = 20  # pixels for one digit, 16+4


n_samples, n_features = X_train.shape
print(X_train.shape)
img_size = int(np.sqrt(n_features))
    

#idx_3 = np.where(y_train == 3)[0]
idx_3 = np.where(y_train == 9)[0]
X_train_3 = X_train[idx_3]


X_train_3_pca = PCA(n_components=2).fit_transform(X_train_3)
x_grid = np.percentile(X_train_3_pca[:, 0], [5, 25, 50, 75, 95])
y_grid = np.percentile(X_train_3_pca[:, 1], [5, 25, 50, 75, 95])
x_grid[2], y_grid[2] = 0, 0

      
        
fig, axarr = plt.subplots(1, 2, figsize=(6.7, 3.8), dpi=150,
                          gridspec_kw=dict(width_ratios=[3, 2]))
plt.subplots_adjust(wspace=0.1)
for s in axarr[1].spines.values():
    s.set_visible(False)
axarr[1].tick_params(
    bottom=False, labelbottom=False, left=False, labelleft=False)
ax = axarr[0]
ax.scatter(X_train_3_pca[:, 0], X_train_3_pca[:, 1], s=1, color='#02A4A3')
ax.set_xlabel('First Principal Component', color=GRAY4, fontsize=8)
ax.set_ylabel('Second Principal Component', color=GRAY4, fontsize=8)
for i in ax.get_yticklabels() + ax.get_xticklabels():
    i.set_fontsize(7)
ax.axhline(0, linewidth=0.5, color=GRAY1)
ax.axvline(0, linewidth=0.5, color=GRAY1)
for i in range(5):
    if i != 2:
        ax.axhline(y_grid[i], linewidth=0.5, color=GRAY4, linestyle='--')
        ax.axvline(x_grid[i], linewidth=0.5, color=GRAY4, linestyle='--')

img = np.ones(shape=(4+w*5, 4+w*5))
for i in range(5):
    for j in range(5):
        v = X_train_3_pca - np.array([x_grid[i], y_grid[j]])
        v = np.sqrt(np.sum(v**2, axis=-1))
        idx = np.argmin(v)
        ax.scatter(
            X_train_3_pca[idx:idx+1, 0], X_train_3_pca[idx:idx+1, 1], s=14,
            facecolors='none', edgecolors='r', linewidth=1)
        img[j*w+4:j*w+4+img_size, i*w+4:i*w+4+img_size] = -X_train_3[idx].reshape(
            (img_size, img_size))

ax = axarr[1]
ax.imshow(img, cmap="gray")
ax.set_aspect('equal', 'datalim')
plt.tight_layout()
pml.savefig('pca_digits.pdf')
plt.show()


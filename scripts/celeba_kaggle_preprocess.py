# Make small version of celebA dataset from kaggle for quickly training a model.
# We use the Kaggle version of the data, which are  (218, 178, 3)
# Download from https://www.kaggle.com/jessicali9530/celeba-dataset
# We extract the first N images and store as a numpy array.
# We save this using pickle for easy loading.

import os
import numpy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from matplotlib.image import imread, imsave
import zipfile

import tensorflow as tf
#from tensorflow import keras
#import tensorflow_datasets as tfds
print(tf.__version__)



DATA_FOLDER = '/home/murphyk/Data/CelebA/'
IMAGE_FOLDER = '/home/murphyk/Data/CelebA/img_align_celeba/'

#INPUT_DIM = (128,128,3)
INPUT_DIM = (64,64,3)
H, W, C = INPUT_DIM





# Useful pre-processing functions
# https://github.com/google/compare_gan/blob/master/compare_gan/datasets.py
# Note change to API in TF 2.0!
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/image/resize_with_crop_or_pad

def preprocess_celeba_tf(img, crop=True):
    # Crop, resize and scale to [0,1]
     # If input is not square, and we resize to a square, we will 
    # get distortions. So better to take a square crop first..
    if crop:
        img = tf.image.resize_with_crop_or_pad(img, 160, 160)
    img = tf.image.resize(img, [H, W])
    img = tf.cast(img, tf.float32) / 255.0
    img = img.numpy()
    return img

def preprocess_celeba_pil(img):
    # we assume img is a PIL Image object.
    img2 = numpy.array(img.resize((H,W)))
    return img2

from os import listdir
from os.path import isfile, join
from glob import glob
mypath = IMAGE_FOLDER
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
print(len(filenames)) # 202,599

N = 20 # number of files we want
crop_image = False
sz = list(INPUT_DIM)
sz.insert(0, N)
X = np.zeros(sz)
show_image = False

out_folder = os.path.join(DATA_FOLDER, 'celeba_small_H{}_W{}_N{}'.format(H, W, N))
if not os.path.exists(out_folder):
      os.makedirs(out_folder)

for i in range(1, N):
    filename = os.path.join(IMAGE_FOLDER, filenames[i])
    #img = PIL.Image.open(filename)
    img = imread(filename) # numpy array
    #img2 = preprocess_celeba_pil(img)
    img2 = preprocess_celeba_tf(img, crop_image)
    if show_image:
        plt.figure()
        plt.imshow(img2)
        plt.show()
    X[i,:,:,:] = img2
    outname = os.path.join(out_folder, filenames[i])
    imsave(outname, img2)

import shutil
shutil.make_archive(out_folder, 'zip', out_folder)

# Load it back
from urllib.request import urlretrieve
import os
from zipfile import ZipFile

def download(url, file):
    if not os.path.isfile(file):
        print("Download file... " + file + " ...")
        urlretrieve(url,file)
        print("File downloaded")

fname = 'celeba_small_H64_W64_N20.zip'
url = 'https://github.com/probml/pyprobml/blob/master/data/{}'.format(fname)        
download(url, os.path.join('/home/murphyk/Data', fname))


# Save numpy array
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html 
folder = DATA_FOLDER    
#fname = 'celeba_N{}_cropped{}.pkl'.format(N, int(crop_image))
fname = 'celeba_N{}_cropped{}.npy'.format(N, int(crop_image))
with open(os.path.join(folder, fname), 'wb') as f:
     #pickle.dump(X, f)   
     np.save(f, X, allow_pickle=False)

with open(os.path.join(folder, fname), 'rb') as f:
    #XX = pickle.load(f)
    XX = np.load(f)
np.allclose(X, XX)    
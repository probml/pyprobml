# We make a small version of celebA dataset 
# We use the Kaggle version of the data:
# Download from https://www.kaggle.com/jessicali9530/celeba-dataset
# We extract the first N images and store as a numpy array.
# We crop and resize images from 218x173 to 64x64.
# We save directory of images as a zip file.
# We also save a small csv file of attributes to match chosen images.
# We can also save numpy array using pickle.

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


#HOME = '/home/murphyk'
HOME = '/Users/kpmurphy'
DATA_FOLDER = os.path.join(HOME, 'Data/CelebA')
IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'img_align_celeba')

#INPUT_DIM = (128,128,3)
INPUT_DIM = (64,64,3)
H, W, C = INPUT_DIM

# Useful pre-processing functions taken from 
# https://github.com/google/compare_gan/blob/master/compare_gan/datasets.py
# Note that in TF2, the tf.image.image_resize is renamed to tf.image.resize

def preprocess_celeba_tf(img, crop=True):
    # Crop, resize and optionally scale to [0,1]
    # If input is not square, and we resize to a square, we will 
    # get distortions. So we  take a square crop first.
    if crop:
        img = tf.image.resize_with_crop_or_pad(img, 160, 160)
    img = tf.image.resize(img, [H, W])
    img = tf.cast(img, tf.float32) 
    img = img / 255.0
    img = img.numpy() # float32
    return img

#https://auth0.com/blog/image-processing-in-python-with-pillow/#Resizing-Images
# We stick with integers so we can save the images easily as jpg
def preprocess_celeba_pil(img, crop=True):
    # Take (160,160) central crop from 218 x 178
    # box (left, upper, right, lower)
    box = (9, 29, 169, 189)
    #img_np = numpy.array(img) # uint8
    #img_pil2 = Image.fromarray(img_np)
    if crop:
        img = img.crop(box)
    img = img.resize((H,W))
    return img # PIL image

from os import listdir
from os.path import isfile, join
#from glob import glob
mypath = IMAGE_FOLDER
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
print(len(filenames)) # 202,599

import pandas as pd
df_all = pd.read_csv(os.path.join(DATA_FOLDER, 'list_attr_celeba.csv'))

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

attr_names = df_all.columns[1:] # drop 'image_id' column
A = df_all[attr_names].to_numpy() # (202599, 40)
plt.imshow(A[:20,:], cmap='gray') # visualize attributes
plt.show()

N = 40000 # number of files we want to use
df_small = df_all[:N] # we can take first N files since they are shuffled


out_name = 'celeba_small_H{}_W{}_N{}'.format(H, W, N)

df_name = os.path.join(DATA_FOLDER, '{}.csv'.format(out_name))
#df_small.to_csv(df_name, sep='\t', index=False)
df_small.to_csv(df_name, index=False) # uses comma as separator

# Folder to store images
out_folder = os.path.join(DATA_FOLDER, out_name)
if not os.path.exists(out_folder):
      os.makedirs(out_folder)

# We resize the N chosen images, store them in a matrix
# and copy them to out_folder 
images = np.zeros((N, H, W, C), dtype=np.float32) # store all resized images here
for i in range(N):
    #filename = filenames[i]
    filename = df_small.iloc[i]['image_id']
    if (N < 100) | (i % 1000 == 0):
        print('processing {}'.format(filename))
    fname = os.path.join(IMAGE_FOLDER, filename)
    img = PIL.Image.open(fname) # PIL image
    img_small = preprocess_celeba_pil(img) # PIL image
    #img = imread(fname) # numpy array of uint8
    #img_small = preprocess_celeba_tf(img) # np array of floats
    img_small_np = numpy.array(img_small) / 255.0
    images[i,:,:,:] = img_small_np
    outname = os.path.join(out_folder, filename)
    #imsave(outname, img2) # stores floats which causes problems
    img_small.save(outname) # save PIL as jpg

# Zip up the image files
import shutil
shutil.make_archive(out_folder, 'zip', out_folder)
print('made {}.zip'.format(out_folder))

# Manually upload zip and csv files to
#https://github.com/probml/pyprobml/blob/master/data/CelebA

'''
# Load it back
from urllib.request import urlretrieve
import os
#from zipfile import ZipFile


df_new = pd.read_csv(os.path.join('/Users/kpmurphy/github/pyprobml/data', 'celeba_small_H64_W64_N20.csv'))


def download(url, file):
    if not os.path.isfile(file):
        print("Download file... " + file + " ...")
        urlretrieve(url,file)
        print("File downloaded")

fname = '{}.zip'.format(out_name)
url = 'https://github.com/probml/pyprobml/blob/master/data/{}'.format(fname)        
download(url, os.path.join(DATA_FOLDER, fname))
'''


'''
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
'''
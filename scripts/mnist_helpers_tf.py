# Helper functions for DNN demos related to mnist images

import superimport

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"


import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display
import sklearn
from time import time

def get_dataset(FASHION=False):
    if FASHION:
      (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data() 
      class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
      (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data() 
      class_names = [str(x) for x in range(10)]
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels, class_names

def plot_dataset(train_images, train_labels, class_names):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    #save_fig("fashion-mnist-data.pdf")
    plt.show()

def plot_image_and_label(predictions_array, true_label, img, class_names):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  img = np.reshape(img, (28, 28)) # drop any trailing dimension of size 1
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("truth={}, pred={}, score={:2.0f}%".format(
      class_names[true_label],
      class_names[predicted_label],
      100*np.max(predictions_array)),
      color=color)

def plot_label_dist(predictions_array, true_label):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
def find_interesting_test_images(predictions, test_labels):
  # We select the first 9 images plus 6 error images
  pred = np.argmax(predictions, axis=1)
  errors = np.where(pred != test_labels)[0]
  print(errors.shape)
  ndx1 = range(9)
  ndx2 = errors[:6]
  ndx = np.concatenate((ndx1, ndx2))
  return ndx

def plot_interesting_test_results(test_images, test_labels, predictions,
                                  class_names, ndx):
  # Plot some test images, their predicted label, and the true label
  # Color correct predictions in blue, incorrect predictions in red
  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    n = ndx[i]
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image_and_label(predictions[n], test_labels[n], test_images[n],
                         class_names)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_label_dist(predictions[n], test_labels[n])
  plt.show()
  
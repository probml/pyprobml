
import superimport

import numpy as np

#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
import tensorflow as tf

import itertools

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
  
def load_mnist_data_keras(flatten=False):
   # Returns X_train: (60000, 28, 28), X_test: (10000, 28, 28), scaled [0..1] 
  # y_train: (60000,) 0..9 ints, y_test: (10000,)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if flatten: 
      Ntrain, D1, D2 = np.shape(x_train)
      D = D1*D2
      assert D == 784
      Ntest = np.shape(x_test)[0]
      x_train = np.reshape(x_train, (Ntrain, D))
      x_test = np.reshape(x_test, (Ntest, D))
    return x_train, x_test, y_train, y_test
  

    
def zscore_normalize(data):
  return (data - data.mean()) / np.maximum(data.std(), 1e-8)

def min_max_normalize(data):
  return (data - data.min()) / np.maximum(data.max() - data.min(), 1e-8)



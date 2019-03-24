'''Script to download MNIST data.

(x_train, y_train, x_test, y_test) = get_mnist()

Downloads the files locally.

x data is uint8 (0 to 255).
y data is uint8 (0 to 9).

'''

# This code gets from Yann Le Cun's original site.
# Code extracted from
# https://github.com/ieee8023/NeuralNetwork-Examples/blob/master/theano/cnn/lasagne-mnist-small-example.ipynb


def get_mnist():
    import sys
    import gzip
    import os
    import numpy as np

    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        urlretrieve(source + filename, filename)
    
    
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            print("Downloading {:s}".format(filename))
            download(filename)
        with gzip.open(filename, 'rb') as f:
            print("Already downloaded {:s}".format(filename))
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
        data = np.asarray([np.rot90(np.fliplr(x[0])) for x in data])
        data = data.reshape(-1, 1, 28, 28)
        return data
        #return data / np.float32(255)
    
    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    t_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    t_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    # Reformat from (N, 1, 28, 28) to (N, 28, 28)
    x_train = x_train.reshape(60000, 28, 28)
    x_test = x_test.reshape(10000, 28, 28)
    
    return (x_train, t_train, x_test, t_test)
    #return {'x_train': x_train, 'y_train': t_train,
    #        'x_test': x_test, 'y_test': t_test}
            
            

'''
# If you have keras installed, this is the easiest way

import keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() # downloads from AWS
'''

'''
#http://scikit-learn.org/stable/datasets/index.html
#from sklearn.datasets import fetch_mldata 
#mnist = fetch_mldata('MNIST original') #download from mldata.org
# mldata.org seems to be down...
'''

'''
# Patch to workaround mldata.org being down
#https://github.com/scikit-learn/scikit-learn/issues/8588

from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os

def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)
'''


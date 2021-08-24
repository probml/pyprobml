# Based on 
# https://github.com/dnouri/skorch/blob/master/notebooks/MNIST.ipynb

# Code using skorch and pure pytorch:
# https://github.com/dnouri/skorch/blob/master/examples/benchmarks/mnist.py

import superimport

import numpy as np
from time import time


import torch
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier

import tensorflow as tf

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


x_train, x_test, y_train, y_test = load_mnist_data_keras()
X_train = np.reshape(x_train, (-1, 784))
X_test = np.reshape(x_test, (-1, 784))

torch.manual_seed(0);
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_dim = X_train.shape[1]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(y_train))

print(mnist_dim, hidden_dim, output_dim) # (784, 98, 10)

# 1 hidden layer MLP
class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=mnist_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X
      
      
net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=10,
    lr=0.1,
    device=device,
)

time_start = time()
net.fit(X_train, y_train);
print('time spent training {}'.format(time() - time_start))

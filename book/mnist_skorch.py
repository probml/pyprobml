# Based on 
# https://github.com/dnouri/skorch/blob/master/notebooks/MNIST.ipynb

# Code using skorch and pure pytorch:
# https://github.com/dnouri/skorch/blob/master/examples/benchmarks/mnist.py

import numpy as np
from time import time


import torch
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier

from utils import load_mnist_data_keras

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

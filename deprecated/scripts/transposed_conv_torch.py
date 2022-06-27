
# Based on http://d2l.ai/chapter_computer-vision/transposed-conv.html


import superimport

import torch
from torch import nn
import numpy as np

def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i + h, j:j + w] += X[i, j] * K
    return Y

# Example from D2L fig 13.10.1
X = torch.tensor([[0., 1], [2, 3]])
K = torch.tensor([[0., 1], [2, 3]])
Y = trans_conv(X, K)
print(Y)

X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
Y2 = tconv(X)
#print(Y2)
assert torch.allclose(Y, Y2)


'''
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding = 1, bias=False)
tconv.weight.data = K
Y2 = tconv(X)
print('Y2', Y2)
'''


# Transposed Matrix multiplication

K = torch.tensor([[1,2], [3, 4]])

def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)

X = torch.tensor([[0.0, 1], [2, 3]])
Y = trans_conv(X, K)
Y2 = torch.mv(W.T, X.reshape(-1)).reshape(3, 3)
assert torch.allclose(Y, Y2)


# Example from Geron fig 14.27

X = torch.ones((2,3))
K = torch.ones(3,3)
X, K = X.reshape(1, 1, 2, 3), K.reshape(1, 1, 3, 3)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, bias=False)
tconv.weight.data = K
Y2 = tconv(X)
print(Y2.shape)

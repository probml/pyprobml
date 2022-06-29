
# Based on http://d2l.ai/chapter_computer-vision/transposed-conv.html

import superimport

import torch
import numpy as np

#K = torch.tensor([[0, 1], [2, 3]])
K = torch.tensor([[1,2], [3, 4]])

print(K)

def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
print(W)

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.arange(9.0).reshape(3, 3)
Y = corr2d(X, K)
print(Y)

Y2 = torch.mv(W, X.reshape(-1)).reshape(2, 2)
assert np.allclose(Y, Y2)
# Based on http://d2l.ai/chapter_computer-vision/transposed-conv.html


import superimport

import jax
import jax.numpy as jnp


def trans_conv(X, K):
    h, w = K.shape
    Y = jnp.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y = Y.at[i : i + h, j : j + w].add(X[i, j] * K)
    return Y


# Example from D2L fig 13.10.1
X = jnp.array([[0.0, 1], [2, 3]])
K = jnp.array([[0.0, 1], [2, 3]])
Y = trans_conv(X, K)
print(Y)

X_ = X.reshape(1, 1, 2, 2)
K_ = jnp.rot90(K, 2).reshape(1, 1, 2, 2)
Y2 = jax.lax.conv_general_dilated(X_, K_, window_strides=(1, 1), padding=((1, 1), (1, 1)), lhs_dilation=(1, 1))
# print(Y2)
assert jnp.allclose(Y, Y2)


"""
X_ = X.reshape(1, 1, 2, 2)
K_ = jnp.rot90(K, 2).reshape(1, 1, 2, 2)
Y2 = jax.lax.conv_general_dilated(X_, K_, window_strides=(1, 1), padding=((0, 0), (0, 0)), lhs_dilation=(1, 1))
print('Y2', Y2)
"""


# Transposed Matrix multiplication

K = jnp.array([[1, 2], [3, 4]])


def kernel2matrix(K):
    k, W = jnp.zeros(5), jnp.zeros((4, 9))
    k = k.at[:2].set(K[0, :])
    k = k.at[3:5].set(K[1, :])
    W = W.at[0, :5].set(k)
    W = W.at[1, 1:6].set(k)
    W = W.at[2, 3:8].set(k)
    W = W.at[3, 4:].set(k)
    return W


W = kernel2matrix(K)

X = jnp.array([[0.0, 1], [2, 3]])
Y = trans_conv(X, K)
Y2 = jnp.dot(W.T, X.reshape(-1)).reshape(3, 3)
assert jnp.allclose(Y, Y2)


# Example from Geron fig 14.27

X = jnp.ones((2, 3))
K = jnp.ones((3, 3))
X_, K_ = X.reshape(1, 1, 2, 3), jnp.rot90(K, 2).reshape(1, 1, 3, 3)
Y2 = jax.lax.conv_general_dilated(X_, K_, window_strides=(1, 1), padding=((2, 2), (2, 2)), lhs_dilation=(2, 2))
print(Y2)
print(Y2.shape)

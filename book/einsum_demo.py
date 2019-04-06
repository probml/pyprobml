# Illustrate einstein summation
# https://rockt.github.io/2018/04/30/einsum
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html

import numpy as np

np.random.seed(42)
a = np.arange(3)
b = np.arange(3)
A = np.arange(6).reshape(2,3)
B = np.arange(15).reshape(3,5)
S = np.arange(9).reshape(3,3)
T = np.random.randn(2,2,2,2)

## Single argument

# Matrix transpose
assert np.allclose(A.T, np.einsum('ij->ji', A))

# Sum all elements
assert np.allclose(np.sum(A), np.einsum('ij->', A))

# Sum across rows
assert np.allclose(np.sum(A, axis=0), np.einsum('ij->j', A))

# Sum across columns
assert np.allclose(np.sum(A, axis=1), np.einsum('ij->i', A))

# Sum specific axis of tensor
assert np.allclose(np.sum(T, axis=1), np.einsum('ijkl->ikl', T))
assert np.allclose(np.sum(np.sum(T, axis=0), axis=0), np.einsum('ijkl->kl', T))

# repeated indices with one arg extracts diagonals
assert np.allclose(np.diag(S), np.einsum('ii->i', S))
          
# Trace
assert np.allclose(np.trace(S), np.einsum('ii->', S))
          

## Two arguments

# Matrix vector multiplication
assert np.allclose(np.dot(A, b), np.einsum('ik,k->i', A, b))

# Matrix matrix multiplication
assert np.allclose(np.dot(A, B), np.einsum('ik,kj->ij', A, B))
assert np.allclose(np.matmul(A, B), np.einsum('ik,kj->ij', A, B))

# Inner product 
assert np.allclose(np.dot(a, b), np.einsum('i,i->', a, b))
assert np.allclose(np.inner(a, b), np.einsum('i,i->', a, b))

# Outer product
assert np.allclose(np.outer(a, b), np.einsum('i,j->ij', a, b))

# Elementwise product
assert np.allclose(a * a, np.einsum('i,i->i', a, a))
assert np.allclose(A * A, np.einsum('ij,ij->ij', A, A))
assert np.allclose(np.multiply(A, A), np.einsum('ij,ij->ij', A, A))

# Batch matrix multiplication
I= 3; J = 2; K = 5; L = 3;
AA = np.random.randn(I,J,K)
BB = np.random.randn(I,K,L)
# C[ijl] = sum_k A[ijk] B[ikl]
CC = np.zeros((I,J,L))
for i in range(I):
    for j in range(J):
        for l in range(L):
            s = 0
            for k in range(K):
                s += AA[i,j,k] * BB[i,k,l]
            CC[i,j,l] = s
assert np.allclose(CC, np.einsum('ijk,ikl->ijl', AA, BB))

## >2 arguments

# Batch sentence embedding and averaging
N = 2; C = 3; D = 4; K = 5; T = 6;
S = np.random.randn(N, T, K)
W = np.random.randn(K, D)
V = np.random.randn(D, C)
L = np.zeros((N,C))
for n in range(N):
    for c in range(C):
        s = 0
        for d in range(D):
            for k in range(K):
                for t in range(T):
                    s += S[n,t,k] * W[k,d] * V[d,c]
        L[n,c] = s
assert np.allclose(L, np.einsum('ntk,kd,dc->nc', S, W, V))

path = np.einsum_path('ntk,kd,dc->nc', S, W, V, optimize='optimal')[0]
assert np.allclose(L, np.einsum('ntk,kd,dc->nc', S, W, V, optimize=path))

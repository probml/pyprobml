# Illustrate einstein summation
# https://rockt.github.io/2018/04/30/einsum
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html

import superimport

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


import jax.numpy as jnp
path = jnp.einsum_path('ntk,kd,dc->nc', S, W, V, optimize='optimal')[0]
assert np.allclose(L, jnp.einsum('ntk,kd,dc->nc', S, W, V, optimize=path))

# Use full student network from KOller and Friedman
str = 'c,dc,gdi,si,lg,jls,hgj->'
K = 5
cptC = np.random.randn(K)
cptD = np.random.randn(K,K)
cptG = np.random.randn(K,K,K)
cptS = np.random.randn(K,K)
cptL = np.random.randn(K,K)
cptJ = np.random.randn(K,K,K)
cptH = np.random.randn(K,K,K)
cpts = [cptC, cptD, cptG, cptS, cptL, cptJ, cptH]
path_info = np.einsum_path(str, *cpts, optimize='optimal')
print(path_info[0]) # 'einsum_path', (0, 1), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1)]
print(path_info[1])
'''
  Complete contraction:  c,dc,gdi,si,lg,jls,hgj->
         Naive scaling:  8
     Optimized scaling:  4
      Naive FLOP count:  2.734e+06
  Optimized FLOP count:  2.176e+03
   Theoretical speedup:  1256.606
  Largest intermediate:  1.250e+02 elements
--------------------------------------------------------------------------
scaling                  current                                remaining
--------------------------------------------------------------------------
   2                     dc,c->d                    gdi,si,lg,jls,hgj,d->
   3                   d,gdi->gi                       si,lg,jls,hgj,gi->
   3                   gi,si->gs                          lg,jls,hgj,gs->
   3                  gs,lg->gls                            jls,hgj,gls->
   4                 gls,jls->gj                                 hgj,gj->
   3                    gj,hgj->                                       ->
'''

path_info = np.einsum_path(str, *cpts, optimize='greedy')
print(path_info[1])
'''
  Complete contraction:  c,dc,gdi,si,lg,jls,hgj->
         Naive scaling:  8
     Optimized scaling:  5
      Naive FLOP count:  2.734e+06
  Optimized FLOP count:  7.101e+03
   Theoretical speedup:  385.069
  Largest intermediate:  1.250e+02 elements
--------------------------------------------------------------------------
scaling                  current                                remaining
--------------------------------------------------------------------------
   5                hgj,jls->gls                     c,dc,gdi,si,lg,gls->
   3                  gls,lg->gs                         c,dc,gdi,si,gs->
   2                     dc,c->d                            gdi,si,gs,d->
   3                   d,gdi->gi                               si,gs,gi->
   3                   gs,si->gi                                  gi,gi->
   2                     gi,gi->                                       ->
'''
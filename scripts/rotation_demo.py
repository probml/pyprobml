import superimport

import numpy as np

a = (45/180) * np.pi
R = np.array(
        [[np.cos(a), -np.sin(a), 0],
          [np.sin(a), np.cos(a), 0],
          [0, 0, 1]])

print("\nR = ")
print(R)

S = np.diag([1.0, 2.0, 3.0])
A = np.dot(np.dot(R, S), R.T)
print("\nA = ")
print(A)

evals, evecs = np.linalg.eig(A)
idx = np.argsort(np.abs(evals)) # smallest first
U = evecs[:, idx] # sort columns
D = np.diag(evals[idx])

assert np.allclose(np.abs(R), np.abs(U))
assert np.allclose(D, S)
assert np.allclose(A, np.dot(U, np.dot(D, U.T)))

a = -a
RR = np.array([[np.cos(a), -np.sin(a), 0],
     [np.sin(a), np.cos(a), 0],
     [0, 0, 1]])

assert np.allclose(R.T, RR)

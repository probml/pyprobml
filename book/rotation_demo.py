import numpy as np

a = (45/180) * np.pi
R = np.array([[np.cos(a), -np.sin(a), 0],
     [np.sin(a), np.cos(a), 0],
     [0, 0, 1]])

print("\nR = ")
print(R)

A = R.dot(np.diag([1,2,3])).dot(R.T)

print("\nA = ")
print(A)

vals, vecs = np.linalg.eig(A)

print("\nU = ")
print(vecs)

print("\nD = ")
print(np.diag(sorted(vals)))


a = -a
RR = np.array([[np.cos(a), -np.sin(a), 0],
     [np.sin(a), np.cos(a), 0],
     [0, 0, 1]])

print()
print(np.allclose(R.T, RR))
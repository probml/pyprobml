import numpy as np

def matprint(mat, fmt="g"):
    """snippet from https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75"""
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

a = (45/180) * np.pi
R = np.array([[np.cos(a), -np.sin(a), 0],
     [np.sin(a), np.cos(a), 0],
     [0, 0, 1]])

print("\nR = ")
matprint(R)

A = R.dot(np.diag([1,2,3])).dot(R.T)

print("\nA = ")
matprint(A)

vals, vecs = np.linalg.eig(A)

print("\nU = ")
matprint(vecs)

print("\nD = ")
matprint(np.diag(sorted(vals)))


a = -a
RR = np.array([[np.cos(a), -np.sin(a), 0],
     [np.sin(a), np.cos(a), 0],
     [0, 0, 1]])

print()
print(np.allclose(R.T, RR))
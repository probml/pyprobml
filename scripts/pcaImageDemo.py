import superimport

from numpy.linalg import matrix_rank
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from sklearn.decomposition import PCA


np.random.seed(0)
# load the faces (value from 0 to 1)
faces = fetch_olivetti_faces()
img = faces['images']
X = faces['data']
target = faces['target']
y = np.ravel(np.repeat(np.arange(1, 41), 10))

h, w, n = 64, 64, len(img)

val = np.random.choice(n, 16, replace=False)
fig, axs = plt.subplots(4, 4)
fig.suptitle("16 Random Face Images ", fontsize="x-large")
for i in range(16):
    r, c = int(i / 4), i % 4
    axs[r, c].imshow(X[val[i]].reshape(h, w), cmap='gray')
    axs[r, c].axis('off')

fig.savefig("../figures/PcaTrainFaceImages.png")

print('Performing PCA')
mu = np.mean(X, axis=0)
XC = X - mu
pca = PCA()
pca.fit(XC)
V = pca.components_
Z = np.dot(XC, V.T)

fig, axs = plt.subplots(2, 2)
fig.suptitle("PCA on Face Images (Principle components) ", fontsize="x-large")
for i in range(4):
    r, c = int(i / 2), i % 2
    if r == 0 and c == 0:
        # mu plot
        axs[r, c].imshow(mu.reshape(h, w), cmap='gray')
        axs[r, c].axis('off')
        axs[r, c].set_title('Mean')
    else:
        # plots the first three Eigenfaces
        axs[r, c].imshow(V[i - 1].reshape(h, w), cmap='gray')
        axs[r, c].axis('off')
        axs[r, c].set_title('principal Basis {}'.format(i - 1))
fig.savefig("../figures/PrincipalComponentFaceImages.png")

ndx = 125
Ks = [5, 10, 20, matrix_rank(X)]
fig, axs = plt.subplots(2, 2)
fig.suptitle("PCA on Face Images (Reconstructed Images) ", fontsize="x-large")
count = 0
for k in Ks:
    Xrecon = np.dot(Z[np.newaxis, ndx, :k], V[:k, :]) + mu
    r, c = int(count / 2), count % 2
    axs[r, c].imshow(Xrecon.reshape(64, 64), cmap='gray')
    axs[r, c].axis('off')
    axs[r, c].set_title('{} Components'.format(k))
    count += 1
fig.savefig("../figures/PCAReconstructedFaceImages.png")


fig, axs = plt.subplots(1, 1)
fig.suptitle("PCA on Face Images (ReconstructionError) ", fontsize="x-large")
Ks = []
Ks.extend(list(np.arange(0, 10, 1)))
Ks.extend(list(np.arange(10, 50, 5)))
Ks.extend(list(np.arange(50, matrix_rank(X), 25)))
mse = np.zeros(len(Ks))
count = 0
for k in Ks:
    Xrecon = np.dot(Z[:, :k], V[:k, :]) + mu
    err = (Xrecon - X)
    mse[count] = np.sqrt(np.mean(err ** 2))
    count += 1

axs.plot(Ks, mse, '-o')
axs.set_ylabel('MSE')
axs.set_xlabel('K')
fig.savefig("../figures/ReconstructionError.png")

fig, axs = plt.subplots(1, 1)
fig.suptitle("pcaImage Faces (proportion of variance) ", fontsize="x-large")

axs.plot(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_), 'o-')
axs.set_ylabel('proportion of variance')
axs.set_xlabel('K')
fig.savefig("../figures/PCAvariance.png")

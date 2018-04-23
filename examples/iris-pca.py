#https://stackoverflow.com/questions/46866208/how-to-use-an-autoencoder-to-visualize-dimensionality-reduction-python-tenso

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os
#%matplotlib inline

# Set random seeds
np.random.seed(0)

# Load data
iris = load_iris()
# Original Iris : (150,4)
X_iris = iris.data 
# Iris with noise : (150,8)
X_iris_with_noise = np.concatenate([X_iris, np.random.random(size=X_iris.shape)], axis=1).astype(np.float32)
X = X_iris
y = iris.target

# PCA
pca_xy = PCA(n_components=2).fit_transform(X)
with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots()
    ax.scatter(pca_xy[:,0], pca_xy[:,1], c=y, cmap=plt.cm.Set2)
    ax.set_title("PCA on Iris")

plt.savefig(os.path.join('figures', 'iris-pca.pdf'))
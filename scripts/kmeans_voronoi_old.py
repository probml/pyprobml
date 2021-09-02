# K-means clustering in 2d
# Code is based on chapter 9 of 
# https://github.com/ageron/handson-ml2

import superimport

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

if 0:
    blob_centers = np.array(
        [[ 0.2,  2.3],
         [-1.5 ,  2.3],
         [-2.8,  1.8],
         [-2.8,  2.8],
         [-2.8,  1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=2000, centers=blob_centers,
                      cluster_std=blob_std, random_state=7)

if 1:
    # two off-diagonal blobs
    X1, _ = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    # three spherical blobs
    blob_centers = np.array(
        [[ -4,  1],
         [-4 ,  3],
         [-4,  -2]])
    s = 0.5
    blob_std = np.array([s, s, s])
    X2, _ = make_blobs(n_samples=1000, centers=blob_centers,
                      cluster_std=blob_std, random_state=7)
    
    X = np.r_[X1, X2]
    K = 5
    
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    
plt.figure(figsize=(8, 4))
plot_clusters(X)
pml.savefig('kmeans_voronoi_data.pdf')
plt.show()


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
        
K = 4
kmeans = KMeans(n_clusters=K, random_state=42)
y_pred = kmeans.fit_predict(X)

plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
pml.savefig('kmeans_voronoi_output.pdf')
plt.show()


# distance of a set of 4 points to each cluster center
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
d1 = kmeans.transform(X_new)
d2 = np.linalg.norm(np.tile(X_new, (1, K)).reshape(-1, K, 2) - kmeans.cluster_centers_, axis=2)
print(d1)
print(d2)


### Plot iterations of K means

seed = 1
kmeans_iter1 = KMeans(n_clusters=K, init="random", n_init=1,
                     algorithm="full", max_iter=1, random_state=seed)
kmeans_iter2 = KMeans(n_clusters=K, init="random", n_init=1,
                     algorithm="full", max_iter=2, random_state=seed)
kmeans_iter3 = KMeans(n_clusters=K, init="random", n_init=1,
                     algorithm="full", max_iter=3, random_state=seed)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)


plt.figure(figsize=(16, 8))
nr = 2; nc = 2;

plt.subplot(nr, nc, 1)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)
plt.title('update centroids')

plt.subplot(nr, nc, 2)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title('assign data points to clusters')

plt.subplot(nr, nc, 3)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(nr, nc, 4)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

plt.tight_layout()
pml.savefig('kmeans_voronoi_iter.pdf')
plt.show()


### K means variability

def plot_clusterer_comparison(clusterer1, clusterer2, X):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    loss = clusterer1.inertia_
    plt.title('distortion = {:0.2f}'.format(loss, fontsize=14))

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    loss = clusterer2.inertia_
    plt.title('distortion = {:0.2f}'.format(loss, fontsize=14))
    
kmeans_rnd_init1 = KMeans(n_clusters=K, init="random", n_init=1,
                         algorithm="full", random_state=2)
kmeans_rnd_init2 = KMeans(n_clusters=K, init="random", n_init=1,
                         algorithm="full", random_state=3)

plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X)

plt.tight_layout()
pml.savefig('kmeans_voronoi_init.pdf')
plt.show()

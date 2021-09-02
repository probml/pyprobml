
# K-means clustering in 2d
# Code is based on  chapter 9 of 
# https://github.com/ageron/handson-ml2

import superimport

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    geron_data = True

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
    geron_data= False


Ks = range(2,9)
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in Ks]
inertias = [model.inertia_ for model in kmeans_per_k]


plt.figure()
plt.plot(Ks, inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Distortion", fontsize=14)
if geron_data:
    plt.annotate('Elbow',
                 xy=(4, inertias[3]),
                 xytext=(0.55, 0.55),
                 textcoords='figure fraction',
                 fontsize=16,
                 arrowprops=dict(facecolor='black', shrink=0.1)
                )
#plt.axis([1, 8.5, 0, 1300])
plt.tight_layout()
plt.savefig("../figures/kmeans_distortion_vs_k.pdf", dpi=300)
plt.show()



silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k]

plt.figure()
plt.plot(Ks, silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7])
plt.tight_layout()
plt.savefig("../figures/kmeans_silhouette_vs_k.pdf", dpi=300)
plt.show()



##########


from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
 
def plot_silhouette(model, X):
    mu = model.cluster_centers_
    K, D= mu.shape
    y_pred = model.labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)
    silhouette_scores = silhouette_score(X, model.labels_)
    cmap = cm.get_cmap("Pastel2")
    colors = [cmap(i) for i in range(K)]
    padding = len(X) // 30
    pos = padding
    for i in range(K):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()
        color = mpl.cm.Spectral(i / K)
        #color = colors[i]
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                         facecolor=color, edgecolor=color, alpha=0.7)
        pos += len(coeffs) + padding
    score = silhouette_scores
    plt.axvline(x=score, color="red", linestyle="--")
    plt.title("$k={}, score={:0.2f}$".format(K, score), fontsize=16)

for model in kmeans_per_k:
    K, D = model.cluster_centers_.shape
    plt.figure()
    plot_silhouette(model, X)
    fname = f'../figures/kmeans_silhouette_diagram{K}.pdf'
    plt.tight_layout()
    plt.savefig(fname, dpi=300)


##########


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(model, X, resolution=1000):
    mu = model.cluster_centers_
    K, D= mu.shape
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #cmap = [mpl.cm.Spectral( (i / K)) for i in range(K)]
    cmap ="Pastel2"
    #cmap = mpl.cm.Spectral(K) 
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),cmap=cmap)
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(model.cluster_centers_)
    plt.title(f'K={K}')

for model in kmeans_per_k:
    K, D = model.cluster_centers_.shape
    plt.figure()
    plot_decision_boundaries(model, X)
    fname = f'../figures/kmeans_silhouette_dboundaries{K}.pdf'
    plt.tight_layout()
    plt.savefig(fname, dpi=300)


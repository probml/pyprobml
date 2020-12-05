

# K-means clustering for semisupervised learning
# Code is from chapter 9 of 
# https://github.com/ageron/handson-ml2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

import itertools
from scipy import linalg


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from matplotlib.colors import LogNorm

X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

w = gm.weights_
mu = gm.means_
Sigma = gm.covariances_

resolution = 100
grid = np.arange(-10, 10, 1 / resolution)
xx, yy = np.meshgrid(grid, grid)
X_full = np.vstack([xx.ravel(), yy.ravel()]).T

# score_samples is the log pdf
pdf = np.exp(gm.score_samples(X_full))
pdf_probas = pdf * (1 / resolution) ** 2
print('integral of pdf {}'.format(pdf_probas.sum()))




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


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))

    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    # plot decision boundaries
    if 0:
        Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z,
                    linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
        

#color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'darkorange'])
color_iter = itertools.cycle(['r', 'g', 'b'])

def plot_assignment(gm, X):
    #plt.figure(figsize=(8, 4))
    plt.figure()
    plt.scatter(X[:,0], X[:,1])
    y_pred = gm.predict(X)
    K = 3
    for k in range(K):
        color = next(color_iter)
        plt.plot(X[y_pred==k, 0], X[y_pred==k, 1], 'o', color=color)



# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
def plot_gaussian(gm, X):
    Y = gm.predict(X)
    means = gm.means_
    covariances = gm.covariances_
    K, D = means.shape
    if gm.covariance_type == 'tied':
        covariances = np.tile(covariances, (K, 1, 1))
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        if gm.covariance_type == 'spherical':
            covar= covar * np.eye(D)
        if gm.covariance_type == 'diag':
            covar= np.diag(covar)
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)
    
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
 

gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
gm_full.fit(X)
gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.tight_layout()
plt.axis('equal')
plt.savefig('../figures/gmm_2d_data.pdf', dpi=300)
plt.show()
    
plt.figure()
plot_assignment(gm_full, X)
plt.tight_layout()
plt.axis('equal')
plt.savefig('../figures/gmm_2d_clustering.pdf', dpi=300)
plt.show()

def make_plot(gm, X, name):     
    ttl = name
    #plt.figure(figsize=(8, 4))
    plt.figure()
    plot_gaussian_mixture(gm, X)
    fname = f'../figures/gmm_2d_{name}_contours.pdf'
    plt.title(ttl)
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig(fname, dpi=300)
    plt.show()
    
    #plt.figure(figsize=(8, 4))
    plt.figure()
    #plot_assignment(gm, X)
    plot_gaussian(gm, X)
    plt.title(ttl)
    fname = f'../figures/gmm_2d_{name}_assignment.pdf'
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig(fname, dpi=300)
    plt.show()


if 1:
    make_plot(gm_full, X, 'full')
    make_plot(gm_tied, X, 'tied')
    make_plot(gm_spherical, X, 'spherical')
    make_plot(gm_diag, X, 'diag')










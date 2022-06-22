# Compare manifold learning methods
#https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>
# Modified by Kevin Murphy

import superimport

from collections import OrderedDict
from functools import partial
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets, decomposition

from pyprobml_utils import save_fig

# Next line to silence pyflakes. This import is needed.
Axes3D



def run_expt(X, color, expt_name):
    n_neighbors = 10
    n_components = 2
    
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    #fig = plt.figure()
    #fig.suptitle("Manifold Learning with %i points, %i neighbors"
    #             % (1000, n_neighbors), fontsize=14)
    
    # Add 3d scatter plot
    #ax = fig.add_subplot(251, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    ttl = '{}-data'.format(expt_name)
    ax.set_title(ttl)    
    save_fig('{}.pdf'.format(ttl))
    plt.show()
    
    # Set-up manifold methods
    LLE = partial(manifold.LocallyLinearEmbedding,
                  n_neighbors, n_components, eigen_solver='auto')
    
    methods = OrderedDict()
    methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    
    methods['PCA'] = decomposition.TruncatedSVD(n_components=n_components)
    methods['LLE'] = LLE(method='standard')
    #methods['LTSA'] = LLE(method='ltsa')
    #methods['Hessian LLE'] = LLE(method='hessian')
    #methods['Modified LLE'] = LLE(method='modified')

    methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                               n_neighbors=n_neighbors)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                     random_state=0)
    methods['kPCA'] = decomposition.KernelPCA(n_components=n_components, kernel='rbf')
    
    # Plot results
    for i, (label, method) in enumerate(methods.items()):
        t0 = time()
        Y = method.fit_transform(X)
        t1 = time()
        print("%s: %.2g sec" % (label, t1 - t0))
        fig = plt.figure()
        # ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
        ax = fig.add_subplot(111)    
        ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        #ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
        ttl = '{}-{}'.format(expt_name, label)
        ax.set_title(ttl)    
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        save_fig('{}.pdf'.format(ttl))
        plt.show()
    
  
    
    
n_points = 1000
noise_levels = [0, 0.2, 0.5, 1.0]
dataset_name = 'swiss'
for noise_ndx, noise in enumerate(noise_levels):
    expt_name = 'manifold-{}-noise-{}'.format(dataset_name, int(noise*100))
    #X, color = datasets.make_s_curve(n_points, random_state=0)
    X, color = datasets.make_swiss_roll(n_points, noise=noise, random_state=42)
    run_expt(X, color, expt_name)
    
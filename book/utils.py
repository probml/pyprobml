
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap

IMAGES_PATH = "../figures"
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="pdf", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch05/ch05.py#L308
def plot_decision_regions_original(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
  

def plot_decision_regions(X, y, classifier, class_names = None):
    sns.set(style="ticks", color_codes=True)
    fig, ax = plt.subplots()
    markers = ('s', 'x', 'o', '^', 'v')
    #colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    #cmap = ListedColormap(colors[:len(np.unique(y))])
    cmap = ListedColormap(sns.color_palette())

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    npoints = 1000
    X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, npoints),
                           np.linspace(x2_min, x2_max, npoints))
    Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape) # NxN array of ints, 0..C-1
    class_ids = np.unique(y)
    nclasses = len(class_ids)
    colors = sns.color_palette()[0:nclasses]
    levels = np.arange(0, nclasses+1)-0.1 # fills in regions z1 < Z <= z2
    ax.contourf(X1, X2, Z, levels=levels, colors=colors, alpha=0.4)
    ax.set(xlim = (X1.min(), X1.max()))
    ax.set(ylim = (X2.min(), X2.max()))

    # plot raw data
    handles = []
    for idx, cl in enumerate(class_ids):
      color = np.atleast_2d(cmap(idx))
      id = ax.scatter(x=X[y == cl, 0], 
                  y=X[y == cl, 1],
                  alpha=0.6, 
                  c=color,
                  edgecolor='black',
                  marker=markers[idx], 
                  label=cl)
      handles.append(id)
    
    if class_names is not None: 
      ax.legend(handles, class_names, scatterpoints=1)
    return fig, ax

# Instability of decision tree classifier in 2d
# Based on https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb

import superimport

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from matplotlib.colors import ListedColormap

def plot_surface(clf, X, y, xnames, ynames):
    n_classes = 3
    plot_step = 0.02
    markers = [ 'o', 's', '^']
    
    plt.figure()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.xlabel(xnames[0])
    plt.ylabel(xnames[1])

    # we pick a color map to match that used by decision tree graphviz 
    cmap = ListedColormap(['orange', 'green', 'purple']) 
    #cmap = ListedColormap(['blue', 'orange', 'green']) 
    #cmap = ListedColormap(sns.color_palette())
    plot_colors = [cmap(i) for i in range(4)]

    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5) 
    # Plot the training points
    for i, color, marker in zip(range(n_classes), plot_colors, markers):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], label=ynames[i],
                    edgecolor='black', color = color, s=50, cmap=cmap, 
                    marker = marker)
    plt.legend()
    

        
        
# Iris data, original
iris = load_iris()


#ndx = [0, 2] # sepal length, petal length
ndx = [2, 3] # petal lenght and width
X = iris.data[:, ndx] 
y = iris.target
xnames = [iris.feature_names[i] for i in ndx]
ynames = iris.target_names

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

plot_surface(tree_clf, X, y, xnames, ynames)
plt.savefig('../figures/dtree_iris_depth2_original.pdf', dpi=300)
plt.show()



# Iris data, perturbed

# Find widest versicolor
ndx = (y==1)
X1 = X[ndx,1]
xmax = X1.max()
# exclude this point
ndx1 = X[:,1] != xmax
ndx2 = (y==2)
not_widest_versicolor = ndx1 | ndx2
X_tweaked = X[not_widest_versicolor]
y_tweaked = y[not_widest_versicolor]

tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X_tweaked, y_tweaked)

plot_surface(tree_clf_tweaked, X, y, xnames, ynames)

class1 = (y==1)
eq_max = X[:,1]==xmax
ndx_omit = np.where(eq_max & class1)[0]
plt.plot(X[ndx_omit, 0], X[ndx_omit, 1], 'r*', markersize=15)

plt.savefig('../figures/dtree_iris_depth2_omit_data.pdf', dpi=300)
plt.show()

# iris data, rotated

Xs = X
ys = y

angle = np.pi / 2
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Xr = Xs.dot(rotation_matrix)
yr = ys

tree_clf_rot = DecisionTreeClassifier(random_state=42)
tree_clf_rot.fit(Xr, yr)

plot_surface(tree_clf_rot, Xr, yr, xnames, ynames)
plt.savefig('../figures/dtree_iris_depth2_rotated.pdf', dpi=300)
plt.show()


## Ensemble of tree fit to original and tweaked data

if 0:
    from sklearn.ensemble import VotingClassifier
    
    eclf = VotingClassifier(
        estimators=[('orig', tree_clf), ('tweaked', tree_clf_tweaked)],
        voting='hard')
    plot_surface(eclf, X, y, xnames, ynames)
    
from prefit_voting_classifier import PrefitVotingClassifier

eclf = PrefitVotingClassifier(
        estimators=[('orig', tree_clf), ('tweaked', tree_clf_tweaked)],
        voting='soft')
plot_surface(eclf, X, y, xnames, ynames)
plt.savefig('../figures/dtree_iris_depth2_ensemble.pdf', dpi=300)
plt.show()   
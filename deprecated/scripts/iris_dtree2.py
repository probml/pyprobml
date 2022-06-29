
# Based on 
# https://scikit-learn.org/stable/modules/tree.html
#https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html

import superimport

import numpy as np
import matplotlib.pyplot as plt

#import seaborn as sns;
#sns.set(style="ticks", color_codes=True)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree

import pydotplus
import collections
import graphviz 
    
figdir = "../figures"

import os



def plot_tree(clf, filename, xnames, ynames):
    tree.plot_tree(clf) 
    dot_data = tree.export_graphviz(clf, out_file=None, 
                        feature_names=xnames,  
                          class_names=ynames,  
                          filled=False, rounded=True,  
                         special_characters=True,
                         label = 'all', impurity=False) 
    graph = graphviz.Source(dot_data)  
    graph.write_pdf(filename)
 

def plot_surface(clf, X, y, filename, xnames, ynames):
    n_classes = 3
    plot_step = 0.02
    markers = ['s', 'o', '*']
    
    plt.figure()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # we pick a color map to match that used by decision tree graphviz 
    cmap = ListedColormap(['#fafab0','#a0faa0', '#9898ff']) # orange, green, blue/purple    
    cs = plt.contourf(xx, yy, Z, cmap=cmap)    
    plot_colors = "ygb"
    
    plt.xlabel(xnames[0])
    plt.ylabel(xnames[1])
    
    # Plot the training points
    for i, color, marker in zip(range(n_classes), plot_colors, markers):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], label=ynames[i],
                    edgecolor='black', color=color, s=30, cmap=cmap, 
                    marker = marker)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    

def demo():
    iris = load_iris()
    ndx =  [2, 3] #  petal length, petal width
    #ndx = [0,1,2,3]
    X = iris.data[:, ndx]
    y = iris.target
    xnames= [iris.feature_names[i] for i in ndx]
    ynames = iris.target_names
    
    depths = [2, 20]
    for depth in depths:
        clf = tree.DecisionTreeClassifier(random_state=42, max_depth=depth)
        clf = clf.fit(X, y)
        
        fname = os.path.join(figdir, 'iris-dtree-2d-tree-depth{}.pdf'.format(depth))
        plot_tree(clf, fname, xnames, ynames)
        
        fname  = os.path.join(figdir, 'iris-dtree-2d-surface-depth{}.pdf'.format(depth))
        plot_surface(clf, X, y, fname, xnames, ynames)

demo()

# Based on 
# https://scikit-learn.org/stable/modules/tree.html
#https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html

import superimport

import numpy as np
import matplotlib.pyplot as plt

#import seaborn as sns;
#sns.set(style="ticks", color_codes=True)

from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree

import pydotplus
import collections
import graphviz 
import pyprobml_utils as pml
    
figdir = "../figures"

import os



def plot_tree(clf, filename, xnames, ynames):
    tree.plot_tree(clf) 
    
 
    dot_data = tree.export_graphviz(clf, out_file=None, 
                        feature_names=xnames,  
                          class_names=ynames,  
                          filled=True, rounded=True,  
                         special_characters=True,
                         label = 'all', impurity=False) 
    graph = graphviz.Source(dot_data)  
    
     # Change color of nodes
    # https://stackoverflow.com/questions/43214350/color-of-the-node-of-tree-with-graphviz-using-class-names
    graph = pydotplus.graph_from_dot_data(dot_data)
    nodes = graph.get_node_list()
    edges = graph.get_edge_list()
    
    edges = collections.defaultdict(list)
    nodes = graph.get_node_list()
    
    cmap = ListedColormap(['orange', 'green', 'purple']) 
    colors = [cmap(i) for i in range(4)]
    
    #colors = {'lightblue', 'lightyellow', 'forestgreen', 'lightred', 'white')
    #colors = ('red', 'yellow', 'blue', 'white')
    
    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            values = clf.tree_.value[int(node.get_name())][0]
            #color only nodes where only one class is present
            if max(values) == sum(values):    
                node.set_fillcolor(colors[np.argmax(values)])
            #mixed nodes get the default color
            else:
                node.set_fillcolor(colors[-1])
                
    graph.write_png('iris-dtree-2d.png')
    #graph.write_pdf(filename)
 

def plot_surface(clf, X, y, filename, xnames, ynames):
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
    
    #cmap=plt.cm.jet
    #cmap=plt.cm.RdYlBu      
    #cmap = ListedColormap(['orange', 'green', 'purple']) 
    cmap = ListedColormap(['blue', 'orange', 'green'])
     
    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    
    #plot_colors = "ryb"
    #plot_colors = "byg"
    plot_colors = [cmap(i) for i in range(3)]
    
    plt.xlabel(xnames[0])
    plt.ylabel(xnames[1])
    
    # Plot the training points
    for i, color, marker in zip(range(n_classes), plot_colors, markers):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], label=ynames[i],
                    edgecolor='black', color=color, s=50, cmap=cmap, 
                    marker = marker)
    plt.legend()
    pml.savefig(filename)
    plt.show()
    

def demo():
    iris = load_iris()
    ndx =  [0, 2] # sepal length, petal length
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
        #plot_tree(clf, fname, xnames, ynames)
        
        fname  = os.path.join(figdir, 'iris-dtree-2d-surface-depth{}.pdf'.format(depth))
        plot_surface(clf, X, y, fname, xnames, ynames)

demo()

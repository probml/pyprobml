# Overfitting with decision tree classifier in 2d
# Based on https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb

import superimport

import numpy as np
import matplotlib.pyplot as plt
import os


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from matplotlib.colors import ListedColormap

def plot_surface(clf, X, y):
    plot_step = 0.02
    
    plt.figure()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # we pick a color map to match that used by decision tree graphviz 
    cmap = ListedColormap(['orange', 'green', 'purple']) 
    #cmap = ListedColormap(['blue', 'orange', 'green']) 
    #cmap = ListedColormap(sns.color_palette())
    plot_colors = [cmap(i) for i in range(4)]

    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5) 
    plt.scatter(X[:,0], X[:,1])
    plt.legend()
    
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=0)
Xtest, ytest = make_moons(n_samples=1000, noise=0.25, random_state=2)

depths = np.arange(1,10)
scores_train = []
scores_test= []
for i, d in enumerate(depths):
    clf = DecisionTreeClassifier(random_state=42, max_depth=d)
    clf.fit(X, y)
    ypred = clf.predict(X)
    score_train = accuracy_score(y, ypred)
    ypred_test = clf.predict(Xtest)
    score_test = accuracy_score(ytest, ypred_test)
    ttl = 'depth={}, train={:0.2f}, test={:.2f}'.format(d, score_train, score_test)
    #plt.figure()
    #plot_surface(clf, X, y)
    #plt.title(ttl)
    scores_train.append(score_train)
    scores_test.append(score_test)
    
plt.figure()
plt.plot(depths, scores_train, label='train accuracy')
plt.plot(depths, scores_test, label='test accuracy')
plt.xlabel('depth')
plt.ylabel('accuracy')
plt.legend()


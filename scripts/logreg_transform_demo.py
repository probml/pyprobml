#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:08:08 2020

@author: kpmurphy
"""

# Fit logistic regression models to 3 classs 2d data.

import superimport

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as mcol
import os

figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))



def plot_data(X0, X1, y):
    for x0, x1, cls in zip(X0, X1, y):
        colors = ['blue', 'black', 'red']
        markers = ['x', 'o', '*']
        color = colors[int(cls)-1]
        marker = markers[int(cls)-1]
        plt.scatter(x0, x1, marker=marker, color=color)



X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0,
    n_classes=2, n_clusters_per_class=2,
    class_sep=1.0, random_state=1234)


degrees = [1, 2, 6]


for i, degree in enumerate(degrees):
    transformer = PolynomialFeatures(degree)
    name = 'Degree{}'.format(degree)
    XX = transformer.fit_transform(X)[:, 1:]  # skip the first column of 1s
    model = LogisticRegression(C=1.0)
    model = model.fit(XX, y)

    n = 100
    xmin = np.min(X[:,0]); xmax = np.max(X[:,0]); 
    ymin = np.min(X[:,1]); ymax = np.max(X[:,1]);
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid2 = transformer.transform(grid)[:, 1:]
    Z = model.predict(grid2).reshape(xx.shape)
    fig, ax = plt.subplots()
    # uses gray background for black dots
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)
    plot_data(X[:, 0], X[:, 1], y)
    plt.title(name)

    fname = 'logregMulti-{}.png'.format(name)
    save_fig(fname)
    plt.draw()

plt.show()

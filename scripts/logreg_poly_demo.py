#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:08:08 2020

@author: kpmurphy
"""

# Fit logistic regression models to 2d data using polynomial features

import superimport

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as mcol
import os
import pyprobml_utils as pml




def plot_data(ax, X, y, is_train=True):
    X0 = X[:,0]; X1 = X[:,1]
    colors = [ 'red', 'blue']
    if is_train:
        markers = ['x', '*']
    else:
        markers = ['o', 's']
    for x0, x1, cls in zip(X0, X1, y):
        color = colors[int(cls)-1]
        marker = markers[int(cls)-1]
        ax.scatter(x0, x1, marker=marker, color=color)
    ax.set_ylim(-2.75,2.75)        

def plot_predictions(ax, xx, yy, transformer, model):
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid2 = transformer.transform(grid)[:, 1:]
    Z = model.predict(grid2).reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.1)
    #plt.axis('off')
    
def make_data(ntrain, ntest):
    n = ntrain + ntest     
    X, y = make_classification(
        n_samples=n, n_features=2, n_redundant=0,
        n_classes=2, n_clusters_per_class=2,
        class_sep=0.1, random_state=1)
    X0, y0 = make_blobs(n_samples=[n, n], n_features=2,
                      cluster_std=2, random_state=1)
    Xtrain = X[:ntrain, :]; ytrain = y[:ntrain]
    Xtest = X[ntrain:, :]; ytest = y[ntrain:]
    xmin = np.min(X[:,0]); xmax = np.max(X[:,0]); 
    ymin = np.min(X[:,1]); ymax = np.max(X[:,1]);
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, 200))
    return Xtrain, ytrain, Xtest, ytest, xx, yy


ntrain = 50; ntest = 1000;
Xtrain, ytrain, Xtest, ytest, xx, yy  = make_data(ntrain, ntest)


### Try different strngth regularizers
degree = 4
# C =1/lambda, so large C is large variance is small regularization
C_list = np.logspace(0, 5, 7)
#C_list = [1, 10, 100, 200, 500, 1000]
plot_list = C_list
err_train_list = []
err_test_list = []
w_list = []
for i, C in enumerate(C_list):
    transformer = PolynomialFeatures(degree)
    name = 'Reg{:d}-Degree{}'.format(int(C), degree)
    XXtrain = transformer.fit_transform(Xtrain)[:, 1:]  # skip the first column of 1s
    model = LogisticRegression(C=C)
    model = model.fit(XXtrain, ytrain)
    w = model.coef_[0]
    w_list.append(w)
    ytrain_pred = model.predict(XXtrain)
    nerrors_train = np.sum(ytrain_pred != ytrain)
    err_train_list.append(nerrors_train / ntrain)                      
    XXtest = transformer.fit_transform(Xtest)[:, 1:]  # skip the first column of 1s
    ytest_pred = model.predict(XXtest)
    nerrors_test = np.sum(ytest_pred != ytest)
    err_test_list.append(nerrors_test / ntest)
    
    if C in plot_list:
        fig, ax = plt.subplots()
        plot_predictions(ax, xx, yy, transformer, model)
        plot_data(ax, Xtrain, ytrain, is_train=True)
        #plot_data(ax, Xtest, ytest, is_train=False)
        ax.set_title(name)
        fname = 'logreg_poly_surface-{}.png'.format(name)
        pml.save_fig(fname)
        plt.draw()
    

plt.figure()
plt.plot(C_list, err_train_list, 'x-', label='train')
plt.plot(C_list, err_test_list, 'o-', label='test')
plt.legend()
plt.xscale('log')
plt.xlabel('Inverse regularization')
plt.ylabel('error rate')
pml.save_fig('logreg_poly_vs_reg-Degree{}.pdf'.format(degree))
plt.show()

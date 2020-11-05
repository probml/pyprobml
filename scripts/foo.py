#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:48:00 2020

@author: kpmurphy
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as mcol
import os

degree = 4
# C =1/lambda, so large C is large variance is small regularization
C_list = [1e0, 1e4]
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
        save_fig(fname)
        plt.draw()
    

plt.figure()
plt.plot(C_list, err_train_list, 'x-', label='train')
plt.plot(C_list, err_test_list, 'o-', label='test')
plt.legend()
plt.xlabel('Inverse regularization')
plt.ylabel('error rate')
save_fig('logreg_poly_vs_reg.png')
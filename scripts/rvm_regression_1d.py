"""
This code is a python version of
https://github.com/probml/pmtk3/blob/master/demos/svmRegrDemo.m
This file performs demos for rbf kernel regressors using L1reg, L2reg, SVM, RVM for noisy sine data
Author: Srikar Reddy Jilugu(@always-newbie161)
"""
import superimport

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.svm import SVR
from cycler import cycler
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import cross_val_score
from rvm_regressor import RelevanceVectorRegressor as RVR
from sklearn.gaussian_process.kernels import RBF
import pyprobml_utils as pml


def main():
    #CB_color = ['#377eb8', '#ff7f00', '#4daf4a']
    #cb_cycler = (cycler(linestyle=['-', '--', '-.']) * cycler(color=CB_color))
    #plt.rc('axes', prop_cycle=cb_cycler)

    # -------------------------------------------
    # making the data
    np.random.seed(0)
    N = 100
    x = 10 * (np.linspace(-1, 1, 100).reshape(-1, 1))
    ytrue = np.array([math.sin(abs(el)) / (abs(el)) for el in x]).reshape(-1, 1)
    noise = 0.1
    y = ytrue + noise * np.random.randn(N, 1)
    X = (x - x.mean()) / x.std()  # normalizing.

    lambd_l2 = 0.1  # regularization parameter for L2reg
    lambd_l1 = 1e-3  # regularization parameter for L1reg
    rbf_scale = 0.3
    gamma = 1 / (2 * rbf_scale ** 2)

    xtest = np.arange(-10, 10.1, 0.1)
    Xtest = (xtest - xtest.mean()) / xtest.std()
    Xtest = Xtest.reshape((-1, 1))

    # applying the rbf kernel feature scaling
    rbf_features = RBFSampler(gamma=gamma, random_state=1)
    rbf_X = rbf_features.fit_transform(X)
    rbf_Xtest = rbf_features.fit_transform(Xtest)

    # -------------------------------------------
    # l2
    reg = linear_model.Ridge(alpha=lambd_l2, fit_intercept=False).fit(rbf_X, y)
    ypred = reg.predict(rbf_Xtest)

    plt.figure()
    plt.plot(X, y, '*')
    plt.plot(Xtest, ypred, '-', color='blue')
    plt.title('linregL2')
    pml.savefig('rvm_data_l2.pdf')

    plt.figure()
    # stem plot of weight vectors.
    plt.title('linregL2')
    plt.stem(reg.coef_.ravel(), use_line_collection=True)
    plt.tight_layout()
    pml.savefig('rvm_stem_l2.pdf')

    # -------------------------------------------
    # l1
    reg = linear_model.Lasso(alpha=lambd_l1, fit_intercept=False,tol=1e-3)
    ypred = reg.fit(rbf_X, y).predict(rbf_Xtest)

    plt.figure()
    plt.plot(X, y, '*')
    plt.plot(Xtest, ypred, '-', color='blue')

    # coefficient vectors of l1reg
    SV_idx = (np.abs(reg.coef_) > 1e-5)
    plt.scatter(X[SV_idx], y[SV_idx], s=200, facecolor="none",edgecolor='red')
    plt.title('linregL1')
    pml.savefig('rvm_data_l1.pdf')

    plt.figure()
    # stem plot of weight vectors.
    plt.title('linregL1')
    plt.stem(reg.coef_.ravel(), use_line_collection=True)
    plt.tight_layout()
    pml.savefig('rvm_stem_l1.pdf')

    # -------------------------------------------
    # RVR
    kernel = RBF(0.3)
    reg = RVR(kernel=kernel)
    reg.fit(X, y.ravel())
    ypred = reg.predict(Xtest)[0]

    plt.figure()
    plt.plot(X, y, '*')
    plt.plot(Xtest, ypred, '-', color='blue')
    # support vectors of RVR
    plt.scatter(reg.X, reg.t, s=200, facecolor="none",edgecolor='red')
    plt.title('RVM')
    plt.tight_layout()
    pml.savefig('rvm_data_rvm.pdf')

    plt.figure()
    # stem plot of weight vectors.
    plt.title('RVM')
    plt.stem(reg.mean.ravel(), use_line_collection=True)
    plt.tight_layout()
    pml.savefig('rvm_stem_rvm.pdf')

    # -------------------------------------------
    # SVM
    C = np.arange(10)+1
    crossval_scores = [cross_val_score(SVR(gamma=gamma, C=c),
                                       X, y.ravel(), scoring='neg_mean_squared_error', cv=5).mean() for c in C]
    c_opt = np.argmin(crossval_scores)
    reg = SVR(gamma=gamma, C=c_opt)
    reg.fit(X, y.ravel())
    ypred = reg.predict(Xtest)

    plt.figure()
    plt.plot(X, y, '*')
    plt.plot(Xtest, ypred, '-', color='blue')

    # support vectors of SVR.
    SV_idx = reg.support_
    plt.scatter(X[SV_idx], y[SV_idx], s=200, facecolor="none",edgecolor='red')
    plt.title('SVM')
    plt.tight_layout()
    pml.savefig('rvm_data_svm.pdf')

    plt.figure()
    # stem plot of weight vectors.
    plt.title('SVM')
    plt.stem(reg.dual_coef_.ravel(), use_line_collection=True)
    plt.tight_layout()
    pml.savefig('rvm_stem_svm.pdf')

    # -------------------------------------------
    plt.show()


if __name__ == "__main__":
    main()
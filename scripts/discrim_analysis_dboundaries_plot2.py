# Gaussian discriminant analysis in 2d
# Author: Duane Rich, heavily modified by Kevin Murphy
# Based on matlab code by Kevin Murphy
#https://github.com/probml/pmtk3/blob/master/demos/discrimAnalysisDboundariesDemo.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = '../figures'
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from sklearn.preprocessing import OneHotEncoder

def mvn2d(x, y, u, sigma):
    xx, yy = np.meshgrid(x, y)
    xy = np.c_[xx.ravel(), yy.ravel()]
    sigma_inv = np.linalg.inv(sigma)
    z = np.dot((xy - u), sigma_inv)
    z = np.sum(z * (xy - u), axis=1)
    z = np.exp(-0.5 * z)
    z = z / (2 * np.pi * np.linalg.det(sigma) ** 0.5)
    return z.reshape(xx.shape)

# Each model specifies the means and covariances.
# If the covariances are equal across classes, dboundarioes
# will be linear even if we use QDA

    
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


model1 = ([[1.5, 1.5], [-1.5, -1.5]], 
           [np.eye(2)] * 2)

model2 = ([[1.5, 1.5], [-1.5, -1.5]],  
           [[[1.5, 0], [0, 1]], np.eye(2) * 0.7])

model3 = ([[0, 0], [0, 5], [5, 5]],
           [np.eye(2)] * 3)

Sigma1 = np.array([[4, 1], [1, 2]])
Sigma2 = np.array([[2, 0], [0, 1]])
Sigma3 = np.eye(2)

model4 = ([[0, 0], [0, 4], [4, 4]],
           [Sigma1, Sigma2, Sigma3])

models = [model1, model2, model3, model4]
models = [model4]

ngrid = 200 
n_samples = 30  # number of each class samples
model_names = ('LDA', 'QDA')
np.random.seed(0)

def make_data(u, sigma):
     # generate random points
    x = []  # store sample points
    labels = []  # store class labels
    nclasses  = len(u) # means
    for i in range(nclasses):
        x.append(np.random.multivariate_normal(u[i], sigma[i], n_samples))
        labels.append([i] * n_samples)
    return x, labels

def make_grid(x):
    points = np.vstack(x)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_range = np.linspace(x_min - 1, x_max + 1, ngrid)
    y_range = np.linspace(y_min - 1, y_max + 1, ngrid)
    xx, yy = np.meshgrid(x_range, y_range)
    return xx, yy, x_range, y_range

def plot_dboundaries(xx, yy, z,  z_p):
    plt.pcolormesh(xx, yy, z, alpha=0.1)
    plt.jet()
    nclasses = z_p.shape[1]
    for j in range(nclasses):
        plt.contour(xx, yy, z_p[:, j].reshape(ngrid, ngrid),
                   [0.5], lw=3, colors='k')

def plot_points(x):
    c = 'bgr'
    m = 'xos'
    for i, point in enumerate(x):
        plt.plot(point[:, 0], point[:, 1], c[i] + m[i])
        
def plot_contours(xx, yy, x_range, y_range, u, sigma):
    nclasses = len(u)
    c = 'bgr'
    m = 'xos'
    for i in range(nclasses):
        prob = mvn2d(x_range, y_range, u[i], sigma[i])
        cs = plt.contour(xx, yy, prob, colors=c[i])
        

def make_one_hot(yhat):
    yy = yhat.reshape(-1,1) # make 2d
    enc = OneHotEncoder(sparse=False)
    Y  = enc.fit_transform(yy)
    return Y

for u, sigma in models:
    x, labels = make_data(u, sigma)
    xx, yy, x_range, y_range = make_grid(x)
    X = np.vstack(x)
    Y = np.hstack(labels)
    
    plt.figure()
    plot_points(x)
    plt.axis('square')
    plt.tight_layout()
    save_fig('gda_2d_data.pdf')
    plt.show()
        
    plt.figure()
    plot_points(x)
    plot_contours(xx, yy, x_range, y_range, u, sigma)
    plt.axis('square')
    plt.tight_layout()
    save_fig('gda_2d_contours.pdf')
    plt.show()
    
    for k, clf in enumerate((LDA(), QDA())):
        clf.fit(X, Y)
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(ngrid, ngrid)
        z_p = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        yhat = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Yhat = make_one_hot(yhat)
        
        plt.figure()
        #plot_dboundaries(xx, yy, z, z_p)
        plot_dboundaries(xx, yy, z, Yhat)
        plot_points(x)
        plot_contours(xx, yy, x_range, y_range, u, sigma)
        plt.title(model_names[k])
        plt.axis('square')
        plt.tight_layout()
        save_fig('gda_2d_{}.pdf'.format(model_names[k]))
        plt.show()






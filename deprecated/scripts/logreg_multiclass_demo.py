# Fit logistic regression models to 3 classs 2d data.

import superimport

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as mcol
import pyprobml_utils as pml

def create_data(N):
    np.random.seed(234)
    # np.random.RandomState(0)
    C = 0.01*np.eye(2)
    Gs = [mvn(mean=[0.5, 0.5], cov=C),
          mvn(mean=[-0.5, -0.5], cov=C),
          mvn(mean=[0.5, -0.5], cov=C),
          mvn(mean=[-0.5, 0.5], cov=C),
          mvn(mean=[0, 0], cov=C)]
    X = np.concatenate([G.rvs(size=N) for G in Gs])
    y = np.concatenate((1*np.ones(N), 1*np.ones(N),
                        2*np.ones(N), 2*np.ones(N),
                        3*np.ones(N)))
    return X, y


def plotScatter(X0, X1, y):
    for x0, x1, cls in zip(X0, X1, y):
        colors = ['blue', 'black', 'red']
        markers = ['x', 'o', '*']
        color = colors[int(cls)-1]
        marker = markers[int(cls)-1]
        plt.scatter(x0, x1, marker=marker, color=color)


X, y = create_data(100)
nclasses = len(np.unique(y))

models = [LogisticRegression(C=1.0),
          LogisticRegression(C=1.0)]

transformers = [PolynomialFeatures(1),  # no-op
                PolynomialFeatures(2)]

names = ['Linear Logistic Regression',
         'Quadratic Logistic Regression']
file_names = ['Linear', 'Quad']

file_type = '.png'  # smaller

for i in range(len(models)):
    transformer = transformers[i]
    XX = transformer.fit_transform(X)[:, 1:]  # skip the first column of 1s
    model = models[i].fit(XX, y)
    #print('experiment %d' % (i))

    #xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
    xx, yy = np.meshgrid(np.linspace(-1, 1, 250), np.linspace(-1, 1, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid2 = transformer.transform(grid)[:, 1:]
    Z = model.predict(grid2).reshape(xx.shape)
    fig, ax = plt.subplots()
    # uses gray background for black dots
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)

    # https://stackoverflow.com/questions/40601997/setting-discrete-colormap-corresponding-to-specific-data-range-in-matplotlib
    #cmap = plt.cm.get_cmap("jet", lut=nclasses)
    #cmap_bounds = np.arange(nclasses+1) - 0.5
    #norm = mcol.BoundaryNorm(cmap_bounds, cmap.N)
    #plt.pcolormesh(xx, yy, Z, cmap=cmap, norm=norm)

    plotScatter(X[:, 0], X[:, 1], y)
    #plt.scatter(X[:,0], X[:,1], y)
    plt.title(names[i])
    plt.draw()
    pml.savefig('logregMulti{}Boundary.png'.format(file_names[i]))
    
plt.show()

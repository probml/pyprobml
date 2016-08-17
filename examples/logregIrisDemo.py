#!/usr/bin/env python

# 3-class Logistic regression on Iris data
# Modified from http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
# The original code uses the default one-vs-rest method (and liblinear solver).
# We change it to softmax loss (and LBFGS solver).

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets

# import the data 
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

logreg = linear_model.LogisticRegression(C=1e5, multi_class='multinomial', 
    solver='lbfgs')
logreg.fit(X, Y)

# Compute predictions on a dense 2d array of inputs. 
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()] # 39501 x 2
Z = logreg.predict(X)
probs = logreg.predict_proba(X) # 39501 x 3

# Plot decision boundary
Z = Z.reshape(xx.shape) # 171 x 231
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('Iris dataset')

# Plot also the training points
markers = ['o', 'x', 's'];
for c in range(3):
    ndx = Y==c
    plt.scatter(X[ndx, 0], X[ndx, 1], marker=markers[c], s=40, c='k', cmap=plt.cm.Paired)
plt.show()

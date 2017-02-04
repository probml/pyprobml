from utils import util
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import os
import random

data = util.load_mat('heightWeight/heightWeight')

random.seed(0)

#Number of mixture components, can be changed, but must be less than 7 (for graphing reasons)
NComponents = 2
MaxIter = 6

X = data['heightWeightData'][:, [1, 2]].astype(np.float64)

#First graph, just to graph the data.
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], marker='o', s=100, facecolors='none', edgecolors='b')
plt.draw()
plt.savefig(os.path.join('figures', 'heightWeightScatter.pdf'))

#Function to standardize the columns of an numpy array.
def standardizeCols(X):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
    return X

X = standardizeCols(X)

#Guassian Mixture Estimator - we fit it and extracts values for graphing.
GMEstimator = GaussianMixture(n_components=NComponents, covariance_type='full', max_iter=MaxIter, random_state=0)
GMFit = GMEstimator.fit(X)
Means = GMFit.means_
Covariances = GMFit.covariances_
Labels = GMFit.predict(X)
LogLike = GMFit.score(X)*X.shape[0] #Because the score returned here is an average score. We want the total.

#Values for drawing the covariance ellipses
n = 100
t = np.linspace(0, 2*np.pi, n)
xy = np.array((np.cos(t), np.sin(t)))
k = np.sqrt(5.99) #This is the scaling factor to go from eigenvectors to a 95 confidence region.

colors = ['r', 'b', 'g', 'y', 'k', 'w', 'c']

#Plotting the scatter plot with confidence regions.
fig, ax = plt.subplots()
for i in range(NComponents):
    [D, U] = np.linalg.eig(Covariances[i])
    wScale = k * np.dot(np.diag(np.sqrt(D)), xy)
    w = np.dot(U, wScale) + np.outer(GMFit.means_[i], np.ones(n))
    ax.scatter(X[:, 0][Labels == i], X[:, 1][Labels == i], marker='o', s=100, facecolors='none', edgecolors=colors[i])
    ax.scatter(GMFit.means_[i][0],GMFit.means_[i][1],marker='x', s=300, color=colors[i])
    ax.plot(w[0, :], w[1, :], color=colors[i])

plt.title('iteration ' + str(MaxIter) + ", loglik " + str(np.round(LogLike, 4)))

plt.savefig(os.path.join('figures', 'heightWeightmixGaussDemo.pdf'))
plt.show()

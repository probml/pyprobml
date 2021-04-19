# -*- coding: utf-8 -*-
#pcaOverfitDemo.py


# Probabilistic PCA
# Profile likelihood of PCA model shows “knee” or “elbow” in the curve

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from tensorflow import keras
import warnings

warnings.simplefilter('ignore', RuntimeWarning) #for some NaN values

# Function to calculate log likelihood of PCA from eigenvalues
# Implemented equations from the book "Probabilistic Machine Learning: An Introduction"

def log_likelihood(evals):

  Lmax = len(evals) 
  ll = np.arange(0.0,Lmax)
  
  for L in range(Lmax):
     
    group1 = evals[0:L+1] #Divide Eigenvalues in two groups
    group2 = evals[L+1:Lmax]

    mu1 = np.mean(group1)
    mu2 = np.mean(group2)

    sigma = (np.sum((group1-mu1)**2 ) + np.sum((group2-mu2)**2)) / Lmax #eqn (20.30)
    
    ll_group1 = np.sum(np.log(multivariate_normal.pdf(group1, mu1, sigma))) 
    ll_group2 = np.sum(np.log(multivariate_normal.pdf(group2, mu2, sigma)))

    ll[L] = ll_group1 + ll_group2 #eqn (20.31)
  
  return ll

# Standard mnist dataset

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images[train_labels == 3] # select 500 images of digit '3'
n_images = 500
train_images = train_images[0:n_images,:,:]
n_samples, n_rows, n_cols = train_images.shape

X_train = np.reshape(train_images, (n_samples, n_rows*n_cols))
n_samples, n_features = X_train.shape
Kmax = min(n_samples, n_features) 

pca = PCA(n_components = Kmax)
X_transformed = pca.fit_transform(X_train)
evals = pca.explained_variance_ #eigenvalues in descending order

ll = log_likelihood(evals)

#Figure 20.8

fig, ax = plt.subplots()
xs = np.arange(1, 51)
ys = ll[0:50]
plt.xlabel('num PCs')
plt.ylabel('profile log likelihood')

ax.plot(xs, ys)
plt.show()
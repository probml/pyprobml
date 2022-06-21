
# Figure 20.6, 20.7, 20.8

# PCA train set and test set reconstruction error vs K
# Reconstruction error on test set gets lower as K increased
# Screeplot and fraction of variance explained
# likelihood of PCA model shows “knee” or “elbow” in the curve

import superimport

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from tensorflow import keras
import warnings
from sklearn.metrics import mean_squared_error

warnings.simplefilter('ignore', RuntimeWarning) #for some NaN values

# Function to calculate log likelihood of PCA from eigenvalues
# Implemented equations from the book:
# "Probabilistic Machine Learning: An Introduction"

def log_likelihood(evals):

  Lmax = len(evals) 
  ll = np.arange(0.0,Lmax)
  
  for L in range(Lmax):
     
    group1 = evals[0:L+1] #Divide Eigenvalues in two groups
    group2 = evals[L+1:Lmax]

    mu1 = np.mean(group1)
    mu2 = np.mean(group2)

    # eqn (20.30)
    sigma = (np.sum((group1-mu1)**2 ) + np.sum((group2-mu2)**2)) / Lmax 
    
    ll_group1 = np.sum(multivariate_normal.logpdf(group1, mu1, sigma))
    ll_group2 = np.sum(multivariate_normal.logpdf(group2, mu2, sigma))

    ll[L] = ll_group1 + ll_group2 #eqn (20.31)
  
  return ll

# Standard mnist dataset

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images[train_labels == 3] # select images of digit '3'

n_images = 1000
train_images = train_images[0:n_images,:,:]
n_samples, n_rows, n_cols = train_images.shape
X = np.reshape(train_images, (n_samples, n_rows*n_cols))

X_train = X[0:int(n_images/2),:] #500 images in train set 
X_test = X[int(n_images/2): ,:] #500 images in test set

#Reconstruction error on MNIST vs number of latent dimensions used by PCA

X_rank = np.linalg.matrix_rank(X_train)
K_linspace = np.linspace(1, 0.75*X_rank, 10, dtype=int)
Ks = np.unique(np.append([1, 5, 10, 20], K_linspace))

RMSE_train= np.arange(len(Ks))
RMSE_test = np.arange(len(Ks))

for index,K in enumerate(Ks):
   pca = PCA(n_components = K)

   Xtrain_transformed = pca.fit_transform(X_train)
   Xtrain_proj = pca.inverse_transform(Xtrain_transformed)
   RMSE_train[index] = mean_squared_error(X_train, Xtrain_proj, squared=False)

   Xtest_transformed = pca.transform(X_test)
   Xtest_proj = pca.inverse_transform(Xtest_transformed)
   RMSE_test[index] = mean_squared_error(X_test, Xtest_proj, squared=False)

#profile log likelihood for PCA

n_samples, n_features = X_train.shape
Kmax = min(n_samples, n_features) 

pca = PCA(n_components = Kmax)
X_transformed = pca.fit_transform(X_train)
evals = pca.explained_variance_ #eigenvalues in descending order

ll = log_likelihood(evals)

#Fraction of variance explained

fraction_var = np.cumsum(evals[0:50]/np.sum(evals))

#Figure 20.6(a) train set reconstruction error

fig, ax = plt.subplots()
xs = Ks
ys = RMSE_train
plt.title('train set reconstruction error')
plt.xlabel('num PCs')
plt.ylabel('rmse')
ax.plot(xs, ys, marker = 'o')
plt.show()

#Figure 20.6(b) test set reconstruction error

fig, ax = plt.subplots()
xs = Ks
ys = RMSE_test
plt.title('test set reconstruction error')
plt.xlabel('num PCs')
plt.ylabel('rmse')
ax.plot(xs, ys, marker = 'o')
plt.show()

#Figure 20.7(a) Scree plot for training set

fig, ax = plt.subplots()
xs = np.arange(1, 51)
ys = evals[0:50]
plt.title('screeplot')
plt.xlabel('num PCs')
plt.ylabel('eigenvalues')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.plot(xs, ys)
plt.show()

#Figure 20.7(b) Fraction of variance explained

fig, ax = plt.subplots()
xs = np.arange(1, 51)
ys = fraction_var
plt.xlabel('num PCs')
plt.ylabel('proportion of variance explained')
ax.plot(xs, ys)
plt.show()

#Figure 20.8 Profile likelihood corresponding to PCA model

fig, ax = plt.subplots()
xs = np.arange(1, 51)
ys = ll[0:50]
plt.xlabel('num PCs')
plt.ylabel('profile log likelihood')

ax.plot(xs, ys)
plt.show()
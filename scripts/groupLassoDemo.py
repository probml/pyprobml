# Author: Meduri Venkata Shivaditya
"""
Figure 11.16 and 11.17 in the book "Probabilistic Machine Learning: An Introduction by Kevin P. Murphy"
Dependencies: spams(pip install spams), group-lasso(pip install group-lasso)
Illustration of group lasso:
To show the effectiveness of group lasso, in this code we demonstrate:
a)Actual Data b)Vanilla Lasso c)Group lasso(L2 norm) d)Group Lasso(L infinity norm)
on signal which is piecewise gaussian and on signal which is piecewise constant
we apply the regression methods to the linear model - y = XW + ε and estimate and plot W
(X)Data: 1024(rows) x 4096(dimensions)
(W)Coefficients : 4096(dimensions)x1(coefficient for the corresponding row)
(ε)Noise(simulated via  N(0,1e-4)): 4096(dimensions) x 1(Noise for the corresponding row)
(y)Target Variable: 1024(rows) x 1(dimension) 

##### Debiasing step #####

Lasso Regression estimator is prone to biasing
Large coefficients are shrunk towards zero
This is why lasso stands for “least absolute selection and shrinkage operator”
A simple solution to the biased estimate problem, known as debiasing, is to use a two-stage
estimation process: we first estimate the support of the weight vector (i.e., identify which elements
are non-zero) using lasso; we then re-estimate the chosen coefficients using least squares.

Sec. 11.5.3. in the book "Probabilistic Machine Learning: An Introduction by Kevin P. Murphy"
for more information

"""
import superimport # installs packages as needed
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg
from group_lasso import GroupLasso
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import spams
from scipy.linalg import lstsq
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(0)

def generate_data(signal_type):
  """
  Generate X, Y and ε for the linear model y = XW + ε
  """
  dim = 2**12
  rows = 2**10
  n_active = 8
  n_groups = 64
  size_groups = dim/n_groups
  #Selecting 8 groups randomly
  rand_perm = np.random.permutation(n_groups)  
  actives = rand_perm[:n_active] 
  groups = np.ceil(np.transpose(np.arange(dim)+1)/size_groups) #Group number for each column 
  #Generating W actual
  W = np.zeros((dim, 1))
  if (signal_type == 'piecewise_gaussian'):
    for i in range(n_active):
      W[groups==actives[i]] = np.random.randn(len(W[groups==actives[i]]),1)
  elif (signal_type == 'piecewise_constant'):
    for i in range(n_active):
      W[groups==actives[i]] =  np.ones((len(W[groups==actives[i]]),1))
  X = np.random.randn(rows, dim)
  sigma = 0.02
  Y = np.dot(X,W) + sigma*np.random.randn(rows,1) #y = XW + ε
  return X,Y,W,groups

def groupLasso_demo(signal_type, fig_start):
  X,Y,W_actual,groups = generate_data(signal_type)
  #Plotting the actual W
  plt.figure(0+fig_start)
  plt.plot(W_actual)
  plt.title("Original (D = 4096, number groups = 64, active groups = 8)")
  plt.savefig("W_actual_{}.png".format(signal_type) , dpi=300)
  ##### Applying Lasso Regression #####
  # L1 norm is the sum of absolute values of coefficients
  lasso_reg = linear_model.Lasso(alpha=0.5)
  lasso_reg.fit(X, Y)
  W_lasso_reg = lasso_reg.coef_
  ##### Debiasing step #####
  ba = np.argwhere(W_lasso_reg != 0) #Finding where the coefficients are not zero
  X_debiased = X[:, ba]
  W_lasso_reg_debiased = np.linalg.lstsq(X_debiased[:,:,0],Y) #Re-estimate the chosen coefficients using least squares
  W_lasso_reg_debiased_2 = np.zeros((4096))
  W_lasso_reg_debiased_2[ba] = W_lasso_reg_debiased[0]
  lasso_reg_mse = mean_squared_error(W_actual, W_lasso_reg_debiased_2)
  plt.figure(1+fig_start)
  plt.plot(W_lasso_reg_debiased_2)
  plt.title('Standard L1 (debiased 1, regularization param(L1 = 0.5), MSE = {:.4f})'.format(lasso_reg_mse))
  plt.savefig("W_lasso_reg_{}.png".format(signal_type), dpi=300)
  ##### Applying Group Lasso L2 regression #####
  # L2 norm is the square root of sum of squares of coefficients 
  # PNLL(W) = NLL(W) + regularization_parameter * Σ(groups)L2-norm
  group_lassoL2_reg = GroupLasso(
    groups=groups,
    group_reg=3,
    l1_reg=1,
    frobenius_lipschitz=True,
    scale_reg="inverse_group_size",
    subsampling_scheme=1,
    supress_warning=True,
    n_iter=1000,
    tol=1e-3,
  )
  group_lassoL2_reg.fit(X, Y)
  W_groupLassoL2_reg = group_lassoL2_reg.coef_
  ##### Debiasing step #####
  ba = np.argwhere(W_groupLassoL2_reg != 0) #Finding where the coefficients are not zero
  X_debiased = X[:, ba]
  W_group_lassoL2_reg_debiased = np.linalg.lstsq(X_debiased[:,:,0],Y) #Re-estimate the chosen coefficients using least squares
  W_group_lassoL2_reg_debiased_2 = np.zeros((4096))
  W_group_lassoL2_reg_debiased_2[ba] = W_group_lassoL2_reg_debiased[0]
  groupLassoL2_mse = mean_squared_error(W_actual, W_group_lassoL2_reg_debiased_2)
  plt.figure(2+fig_start)
  plt.plot(W_group_lassoL2_reg_debiased_2)
  plt.title('Block-L2 (debiased 1, regularization param(L2 = 3, L1=1), MSE = {:.4f})'.format(groupLassoL2_mse))
  plt.savefig("W_groupLassoL2_reg_{}.png".format(signal_type), dpi=300)
  ##### Applying Group Lasso Linf regression #####
  # To use spams library, it is necessary to convert data to fortran normalized arrays
  # visit http://spams-devel.gforge.inria.fr/ for the documentation of spams library
  # Linf is the supremum of all the coeifficients
  # PNLL(W) = NLL(W) + regularization_parameter * Σ(groups)Linf-norm
  X_normalized = np.asfortranarray(X - np.tile(np.mean(X,0),(X.shape[0],1)),dtype=float)
  X_normalized = spams.normalize(X_normalized)
  Y_normalized = np.asfortranarray(Y - np.tile(np.mean(Y,0),(Y.shape[0],1)),dtype=float)
  Y_normalized = spams.normalize(Y_normalized)
  groups_modified = np.concatenate([[i] for i in groups]).reshape(-1, 1)
  W_initial = np.zeros((X_normalized.shape[1],Y_normalized.shape[1]),dtype=float,order="F")
  param = {'numThreads' : -1,'verbose' : True,
  'lambda2' : 3, 'lambda1' : 1, 'max_it' : 500,
  'L0' : 0.1, 'tol' : 1e-2, 'intercept' : False,
  'pos' : False, 'loss' : 'square'}
  param['regul'] = "group-lasso-linf"
  param2=param.copy()
  param['size_group'] = 64
  param2['groups'] = groups_modified
  (W_groupLassoLinf_reg, optim_info) = spams.fistaFlat(Y_normalized,X_normalized,W_initial,True,**param)
  ##### Debiasing step #####
  ba = np.argwhere(W_groupLassoLinf_reg != 0) #Finding where the coefficients are not zero
  X_debiased = X[:, ba[:,0]]
  W_groupLassoLinf_reg_debiased = np.linalg.lstsq(X_debiased,Y) #Re-estimate the chosen coefficients using least squares
  W_group_lassoLinf_reg_debiased_2 = np.zeros((4096))
  W_group_lassoLinf_reg_debiased_2[ba] = W_groupLassoLinf_reg_debiased[0]
  groupLassoLinf_mse = mean_squared_error(W_actual, W_group_lassoLinf_reg_debiased_2)
  plt.figure(3+fig_start)
  axes = plt.gca()
  plt.plot(W_group_lassoLinf_reg_debiased_2)
  plt.title('Block-Linf (debiased 1, regularization param(L2 = 3, L1=1), MSE = {:.4f})'.format(groupLassoLinf_mse))
  plt.savefig("W_groupLassoLinf_reg_{}.png".format(signal_type), dpi=300)
  plt.show()

def main():
  groupLasso_demo('piecewise_gaussian', fig_start=0)
  groupLasso_demo('piecewise_constant', fig_start=4)

if __name__ == "__main__":
    main()

# Variational Bayes for binary logistic regression
# Written by Amazasp Shaumyan

#https://github.com/AmazaspShumik/sklearn-bayes/blob/master/ipython_notebooks_tutorials/linear_models/bayesian_logistic_regression_demo.ipynb

import superimport

#from skbayes.linear_models import EBLogisticRegression,VBLogisticRegression
from bayes_logistic import EBLogisticRegression, VBLogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from pyprobml_utils import save_fig

from scipy import stats
from matplotlib import cm

# create data set 
np.random.seed(0)
n_samples  = 500
x          = np.random.randn(n_samples,2)
x[0:250,0] = x[0:250,0] + 3
x[0:250,1] = x[0:250,1] - 3
y          = -1*np.ones(500)
y[0:250]   = 1
eblr = EBLogisticRegression(tol_solver = 1e-3)
vblr = VBLogisticRegression()   
eblr.fit(x,y)
vblr.fit(x,y)

# create grid for heatmap
n_grid = 500
max_x      = np.max(x,axis = 0)
min_x      = np.min(x,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))

eblr_grid = eblr.predict_proba(Xgrid)[:,1]
vblr_grid = vblr.predict_proba(Xgrid)[:,1]
grids = [eblr_grid, vblr_grid]
lev   = np.linspace(0,1,11)  
titles = ['Type II Bayesian Logistic Regression', 'Variational Logistic Regression']
for title, grid in zip(titles, grids):
    plt.figure(figsize=(8,6))
    plt.contourf(X1,X2,np.reshape(grid,(n_grid,n_grid)),
                 levels = lev,cmap=cm.coolwarm)
    plt.plot(x[y==-1,0],x[y==-1,1],"bo", markersize = 3)
    plt.plot(x[y==1,0],x[y==1,1],"ro", markersize = 3)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
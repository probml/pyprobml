# Demo of  logistic regression with automatic relevancy determination
# to eliminate irrelevant features.

#https://github.com/AmazaspShumik/sklearn-bayes/blob/master/ipython_notebooks_tutorials/rvm_ard/ard_classification_demo.ipynb

import superimport

from ard_linreg_logreg import ClassificationARD
from ard_vb_linreg_logreg import VBClassificationARD 

import numpy as np
import matplotlib.pyplot as plt
from pyprobml_utils import save_fig

from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


def generate_dataset(n_samples = 500, n_features = 100,
                     cov_class_1 = [[0.9,0.1],[1.5,.2]],
                     cov_class_2 = [[0.9,0.1],[1.5,.2]],
                     mean_class_1 = (-1,0.4),
                     mean_class_2 = (-1,-0.4)):
    ''' Generate binary classification problem with two relevant features'''
    X   = np.random.randn(n_samples, n_features)
    Y   = np.ones(n_samples)
    sep = int(n_samples/2)
    Y[0:sep]     = 0
    X[0:sep,0:2] = np.random.multivariate_normal(mean = mean_class_1, 
                   cov = cov_class_1, size = sep)
    X[sep:n_samples,0:2] = np.random.multivariate_normal(mean = mean_class_2,
                        cov = cov_class_2, size = n_samples - sep)
    return X,Y




    
def run_demo(n_samples, n_features):
    np.random.seed(42)
    X,Y = generate_dataset(n_samples,n_features)
    
    plt.figure(figsize = (8,6))
    plt.plot(X[Y==0,0],X[Y==0,1],"bo", markersize = 3)
    plt.plot(X[Y==1,0],X[Y==1,1],"ro", markersize = 3)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title("Example of dataset")
    plt.show()
    
    # training & test data
    X,x,Y,y = train_test_split(X,Y, test_size = 0.4)
    
    models = list()
    names = list()
    
    models.append(ClassificationARD())
    names.append('logreg-ARD-Laplace')
    
    models.append(VBClassificationARD())
    names.append('logreg-ARD-VB')
    
    models.append(LogisticRegressionCV(penalty = 'l2', cv=3))
    names.append('logreg-CV-L2')
    
    models.append(LogisticRegressionCV(penalty = 'l1', solver = 'liblinear', cv=3))
    names.append('logreg-CV-L1')
    
        
    nmodels = len(models)
    for i in range(nmodels):
        print('\nfitting {}'.format(names[i]))
        models[i].fit(X,Y)
                    
    # construct grid    
    n_grid = 100
    max_x      = np.max(x[:,0:2],axis = 0)
    min_x      = np.min(x[:,0:2],axis = 0)
    X1         = np.linspace(min_x[0],max_x[0],n_grid)
    X2         = np.linspace(min_x[1],max_x[1],n_grid)
    x1,x2      = np.meshgrid(X1,X2)
    Xgrid      = np.zeros([n_grid**2,2])
    Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
    Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
    Xg         = np.random.randn(n_grid**2,n_features)
    Xg[:,0]    = Xgrid[:,0]
    Xg[:,1]    = Xgrid[:,1]
    
    # estimate probabilities for grid data points
    #preds = [0]*nmodels # iniitialize list
    for i in range(nmodels):
        pred = models[i].predict_proba(Xg)[:,1]
        fig,ax = plt.subplots()
        ax.contourf(X1,X2,np.reshape(pred,(n_grid,n_grid)),cmap=cm.coolwarm)
        ax.plot(x[y==0,0],x[y==0,1],"bo", markersize = 5)
        ax.plot(x[y==1,0],x[y==1,1],"ro", markersize = 5)
        nnz = np.sum(models[i].coef_ != 0)
        ax.set_title('method {}, N={}, D={}, nnz {}'.format(names[i], n_samples, n_features, nnz))
        name = '{}-N{}-D{}.pdf'.format(names[i], n_samples, n_features)
        save_fig(name)
        plt.show()
    
ndims = [100]
ndata = [100, 200, 500]
for n_samples in ndata:
    for n_features in ndims:
        run_demo(n_samples, n_features)
import scipy
from scipy.io import loadmat
import random
import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import BernoulliNB,MultinomialNB,GaussianNB
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import os
import pyprobml_utils as pml

!wget 'https://github.com/probml/probml-data/blob/main/data/XwindowsDocData.mat?raw=true' -O XwindowsDocData.mat

data = loadmat('XwindowsDocData.mat')
Xtrain = data['xtrain']
Xtrain = scipy.sparse.csc_matrix.toarray(Xtrain)
Xtest = data['xtest']
Xtest = scipy.sparse.csc_matrix.toarray(Xtest)

ytrain = data['ytrain']
ytest = data['ytest']
model = BernoulliNB()

model.fit(Xtrain, ytrain)

ypred_train = model.predict(Xtrain)

err_train = np.mean(zero_one_loss(ytrain, ypred_train))

ypred_test = model.predict(Xtest)

err_test = np.mean(zero_one_loss(ytest, ypred_test))

print('misclassification rates  on train = '+str(err_train*100) +
      ' pc, on test = '+str(err_test*100)+' pc\n')

C = np.unique(data['ytrain']).size

print()
for i in range(0, C):
    plt.bar(np.arange(0, 600, 1), np.exp(model.feature_log_prob_)[i, :])
    plt.title(r'$P(x_j=1 \mid y='+str(i+1)+')$')
    fileName = 'naiveBayesBow'+str(i+1)+'ClassCond'
    pml.savefig(fileName+'.pdf')
    plt.show()

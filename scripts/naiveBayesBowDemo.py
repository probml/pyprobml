from pymatreader import read_mat
import scipy
import random
import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import os

if os.path.isdir('scripts'):
    os.chdir('scripts')

data = None
Xtrain = None
Xtest = None

data = read_mat('../data/XwindowsDocData.mat')
Xtrain = data['xtrain']
Xtrain = scipy.sparse.csc_matrix.toarray(Xtrain)
Xtest = data['xtest']
Xtest = scipy.sparse.csc_matrix.toarray(Xtest)

ytrain = data['ytrain']
ytest = data['ytest']
model = BernoulliNB()
ypred_train = model.fit(Xtrain, ytrain).predict(Xtrain)
err_train = np.mean(zero_one_loss(ytrain, ypred_train))
ypred_test = model.fit(Xtrain, ytrain).predict(Xtest)
err_test = np.mean(zero_one_loss(ytest, ypred_test))
print('misclassification rates  on train = '+str(err_train*100) +
      ' pc, on test = '+str(err_test*100)+' pc\n')

C = np.unique(data['ytrain']).size

print()
for i in range(0, C):
    plt.bar(np.arange(0, 600, 1), np.exp(model.feature_log_prob_)[i, :])
    plt.title('p(xj=1|y='+str(i)+')')
    fileName = 'naiveBayesBow'+str(i+1)+'ClassCond'
    plt.savefig(r'../figures/'+fileName)
    plt.show()

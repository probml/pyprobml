# Compare various online solvers (SGD, SAG) on various classification
# datasets using multinomial logisti regression with L2 regularizer.

#https://scikit-learn.org/stable/modules/linear_model.html



import superimport

import numpy as np
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
#from lightning.classification import SGDClassifier as SGDClassifierLGT
from sklearn import datasets
from sklearn.model_selection import train_test_split
import time
   
"""
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
#X = iris.data # all 4 features

y = iris.target # multiclass
# BFGS gets 10 errors, SGD/OVO gets 20!

#y = (iris.target==2) # make into a binary problem 
# class 0: both get 0 errors,
# class 1: both get 15 err
# class 2: bfgs gets 8, sgd gets 10
"""


"""
digits = datasets.load_digits()
X, y = digits.data, digits.target # (1797, 64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
max_iter = 20
"""


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
X, y = mnist["data"], mnist["target"]
X = X / 255 # convert to real in [0,1] 
y = y.astype(np.uint8)
print(X.shape) # (70000, 784)
# Standard train/ test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


#https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html

from sklearn.utils import check_random_state
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
train_samples = 10000
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)


scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


max_iter = 10

# L2-regularizer is alpha=1/C for sklearn
# Note that alpha is internally multiplied by 1/N before being used
#https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/linear_model/sag.py#L244


#https://stats.stackexchange.com/questions/216095/how-does-alpha-relate-to-c-in-scikit-learns-sgdclassifier
#1. / C_svr ~ 1. / C_svc ~ 1. / C_logistic ~ alpha_elastic_net * n_samples ~ alpha_lasso * n_samples ~ alpha_sgd * n_samples ~ alpha_ridge


Ntrain = np.shape(X_train)[0]
l2reg = 1e-2
alpha = l2reg * Ntrain
solvers = []
if Ntrain < 2000:
  solvers.append( ('BFGS', LogisticRegression(
      C=1/alpha, penalty='l2', solver='lbfgs', multi_class='multinomial')))
solvers.append( ('SAG', LogisticRegression(
    C=1/alpha, penalty='l2', solver='sag', multi_class='multinomial',
    max_iter = max_iter, tol=1e-1)))
solvers.append( ('SAGA', LogisticRegression(
    C=1/alpha, penalty='l2', solver='saga', multi_class='multinomial',
    max_iter = max_iter, tol=1e-1)))
solvers.append( ('SGD', SGDClassifier(
    loss='log', alpha=alpha, penalty='l2',
    eta0 = 1e-3, 
    learning_rate='adaptive', max_iter=max_iter, tol=1e-1)))
#solvers.append( ('SGD', SGDClassifier(
#    loss='log', alpha=alpha, penalty='l2',
#    learning_rate='optimal', max_iter=max_iter, tol=1e-1)))
#solvers.append( ('SGD-LGT', SGDClassifierLGT(
#    loss='log', alpha=l2reg, penalty='l2',
#    multiclass=True, max_iter=30)))


for name, model in solvers:
  t0 = time.time()
  model.fit(X_train, y_train)
  train_time = time.time() - t0
  train_acc = model.score(X_train, y_train)
  test_acc = model.score(X_test, y_test)
  print("{}: train time {:0.2f}, train acc {:0.2f}, test acc {:0.2f}".format(
      name, train_time,  train_acc, test_acc))


"""
SAG: train time 3.25, train acc 0.89, test acc 0.89
SAGA: train time 4.61, train acc 0.88, test acc 0.88
SGD: train time 5.28, train acc 0.24, test acc 0.25
"""


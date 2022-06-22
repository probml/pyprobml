
# Training code based on 
#https://github.com/scikit-learn/scikit-learn/blob/master/benchmarks/bench_mnist.py

import superimport

import numpy as np
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import zero_one_loss



#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml


def load_mnist_data_openml():
  # Returns X_train: (60000, 784), X_test: (10000, 784), scaled [0...1]
  # y_train: (60000,) 0..9 ints, y_test: (10000,)
    print("Downloading mnist...")
    data = fetch_openml('mnist_784', version=1, cache=True)
    print("Done")
    #data = fetch_mldata('MNIST original')
    X = data['data'].astype('float32')
    y = data["target"].astype('int64')
    # Normalize features
    X = X / 255
    # Create train-test split (as [Joachims, 2006])
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_mnist_data_openml()


max_iter = 10 # 400
ESTIMATORS = {
    'LogReg-SAG': LogisticRegression(solver='sag', tol=1e-1,
                                     multi_class='multinomial', C=1e4),
    'LogReg-SAGA': LogisticRegression(solver='saga', tol=1e-1,
                                      multi_class='multinomial', C=1e4),
    'MLP-SGD': MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=max_iter, alpha=1e-4,
        solver='sgd', learning_rate_init=0.2, momentum=0.9, verbose=1,
        tol=1e-1, random_state=1),
    'MLP-adam': MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=max_iter, alpha=1e-4,
        solver='adam', learning_rate_init=0.001, verbose=1,
        tol=1e-1, random_state=1)
}



print("Training Classifiers")
print("====================")
error, train_time, test_time = {}, {}, {}
names = ESTIMATORS.keys()
for name in names:
      print("Training %s ... " % name, end="")
      estimator = ESTIMATORS[name]
      estimator_params = estimator.get_params()

      time_start = time()
      estimator.fit(X_train, y_train)
      train_time[name] = time() - time_start

      time_start = time()
      y_pred = estimator.predict(X_test)
      test_time[name] = time() - time_start

      error[name] = zero_one_loss(y_test, y_pred)

      print("done")

print()
print("Classification performance:")
print("===========================")
print("{0: <24} {1: >10} {2: >11} {3: >12}"
      "".format("Classifier  ", "train-time", "test-time", "error-rate"))
print("-" * 60)
for name in sorted(names, key=error.get):
    print("{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}"
          "".format(name, train_time[name], test_time[name], error[name]))
print()
"""
10 epochs
===========================
Classifier               train-time   test-time   error-rate
------------------------------------------------------------
MLP-adam                     20.59s       0.09s       0.0232
MLP-SGD                      22.95s       0.10s       0.0246
LogReg-SAGA                  15.12s       0.04s       0.0746
LogReg-SAG                   10.76s       0.03s       0.0749
"""

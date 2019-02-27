
# Visualization Code based on 
#https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb

# Training code based on 
#https://github.com/scikit-learn/scikit-learn/blob/master/benchmarks/bench_mnist.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from time import time
import os
from joblib import Memory
#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
from sklearn.datasets import get_data_home

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import zero_one_loss

from utils import save_fig  

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

  
# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), 'mnist_benchmark_data'),
                mmap_mode='r')

@memory.cache
def load_data():
    """Load the data, then cache and memmap the train/test split"""
    # Load dataset
    print("Loading dataset...")
    data = fetch_openml('mnist_784', version=1, cache=True)
    #data = fetch_mldata('MNIST original')
    #X = check_array(data['data'], dtype=dtype, order=order)
    X = data['data']
    y = data["target"]

    # Normalize features
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test
  
  

X_train, X_test, y_train, y_test = load_data()
 
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    montage = np.concatenate(row_images, axis=0)
    plt.imshow(montage, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    
plt.figure(figsize=(9,9))
example_images = X_train[:100]
plot_digits(example_images, images_per_row=10)
save_fig("mnist_digits")
plt.show()

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

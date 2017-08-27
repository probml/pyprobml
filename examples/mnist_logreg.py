'''Logistic regression classifier on mnist.

Borrows some code from
# http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
# http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
'''


import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import linear_model
from sklearn import metrics
from timeit import default_timer as timer

from examples import get_mnist
(x_train, y_train, x_test, y_test) = get_mnist.get_mnist()
#(x_train, y_train, x_test, y_test) = get_mnist()


'''
# Sanity check
import keras
from keras.datasets import mnist
(x_train_k, y_train_k), (x_test_k, y_test_k) = mnist.load_data()
np.array_equal(x_train, x_train_k)
# x_train[0,-4]
'''

ntrain = x_train.shape[0] #60k
ntest = x_test.shape[0] # 10k
num_classes = len(np.unique(y_train)) # 10
ndims = x_train.shape[1] * x_train.shape[2] # 28*28=784

# Preprocess data
x_train = x_train.reshape(ntrain, ndims)
x_test = x_test.reshape(ntest, ndims)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('starring training')
classifier = linear_model.LogisticRegression(C=1e5, multi_class='multinomial', 
    solver='sag', max_iter=10, verbose=1)
start = timer()
classifier.fit(x_train, y_train)
end = timer()
print('Training took {:f} seconds'.format(end - start))

'''
starring training
max_iter reached after 25 seconds
Training took 25.566608 seconds
Accuracy on test set 0.924600
/Users/kpmurphy/Library/Enthought/Canopy/edm/envs/User/lib/python3.5/site-packages/sklearn/linear_model/sag.py:286: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  "the coef_ did not converge", ConvergenceWarning)
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   25.2s finished
'''

predicted = classifier.predict(x_test)
expected = y_test
acc = metrics.accuracy_score(expected, predicted)
misclassified_ndx = np.argwhere(predicted != expected)
nerrors = len(misclassified_ndx)
print("Performance on test set. Accuracy {:f}, nerrors {:d}".format(acc, nerrors))

# Show first 9 images
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.axis('off')
    j = i 
    #j = int(misclassified_ndx[i])
    image = np.reshape(x_test[j], [28, 28])
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ttl = 'True {:d}, Pred: {:d}'.format(expected[j], predicted[j])
    plt.title(ttl)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join('figures', 'mnist_logreg.pdf'))

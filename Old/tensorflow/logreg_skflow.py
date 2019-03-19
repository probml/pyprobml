
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn.metrics import accuracy_score
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import random_seed

use_mnist = False

if use_mnist:
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels
    n_classes = 10
    batch_size = 32
else:
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    X_train, X_test, y_train, y_test = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    n_classes = 3
    batch_size = X_train.shape[0]
    
seed = 42
random_seed.set_random_seed(seed)


model_fn =  skflow.models.logistic_regression  
classifier = skflow.TensorFlowEstimator(model_fn=model_fn, n_classes=3, 
                batch_size=batch_size)                                 
classifier.fit(X_train, y_train, logdir="/tmp/data/my_model")

# Use this command to visualize results
#tensorboard --logdir=/tmp/data/my_model_1


weights = classifier.get_tensor_value('logistic_regression/weights:0')
bias = classifier.get_tensor_value('logistic_regression/bias:0')
print('weights {}, bias {}'.format(weights, bias))
# weights 
# [[ 0.11711627  0.0507526  -0.6172241 ]
# [ 0.49042985 -0.33068454 -1.21840823]
# [-0.91637373 -0.12130416  1.06682682]
# [-1.72253537  0.19360235  1.08898294]]
# bias [ 1.73875201  0.99043357  0.53929734]


y_pred_train = classifier.predict(X_train, batch_size=X_train.shape[0])
score_train = accuracy_score(y_pred_train, y_train)
print('Train accuracy: %f' % score_train) # 0.975000


y_pred_test = classifier.predict(X_test, batch_size=X_test.shape[0])
score_test = accuracy_score(y_pred_test, y_test)
print('Test accuracy: %f' % score_test) # 0.966667


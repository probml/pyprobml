"""MNIST classifier using logistic regression.
Based on github/tensorflow/tensorflow/examples/tutorials/mnist/mnist_softmax.py
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn.metrics import accuracy_score
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split
from tensorflow.examples.tutorials.mnist import input_data



use_mnist = False

if use_mnist:
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels
    n_classes = 10
    n_features = 28*28 # each imnage is 28*28 pixels
else:
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    X_train, X_test, y_train, y_test = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    n_classes = 3


# Create the model
X = tf.placeholder(tf.float32, [None, n_features])
W = tf.Variable(tf.zeros([n_features, n_features]))
b = tf.Variable(tf.zeros([n_classes]))
logits = tf.matmul(X, W) + b
y_pred = tf.nn.softmax(logits)


# Define loss
#y_true = tf.placeholder(tf.float32, [None, n_classes]) # one hot
#cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred))
y_true = tf.placeholder(tf.int64, [None])
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_true)

# Trainer
train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Eval
correct_prediction = tf.equal(tf.argmax(y_pred, 1), y_true)
#correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Fit model
init_op = tf.initialize_all_variables()

# Run!
sess = tf.InteractiveSession()
init_op.run()
for i in range(200):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # batch_ys has shape  B * 10, where each row is one-hot encoded
  if i % 100 == 0:
      print("step {}".format(i))
  train_step.run({X: batch_xs, y_true: batch_ys})

# Test trained model
test_accuracy = accuracy.eval({X: mnist.test.images, y_true: mnist.test.labels})
print("test accuracy {0:0.2f}".format(test_accuracy))
assert np.isclose(test_accuracy, 0.91, 1e-1), (
    'test accuracy should be 0.91, is %f' % test_accuracy)




import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn.metrics import accuracy_score
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.io.data_feeder import setup_train_data_feeder
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.python.ops import nn
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.io.data_feeder import setup_train_data_feeder
from tensorflow.contrib.layers import optimizers

use_mnist = False

if use_mnist:
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels
    batch_size = 32
    n_epochs = 200
    lr = 0.1
else:
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    X_train, X_test, y_train, y_test = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    batch_size = X_train.shape[0]
    # copy the skflow defaults
    n_epochs = 200
    lr = 0.1
    
Y_train_mat = np.reshape(y_train, (len(y_train), 1))
enc = OneHotEncoder()
enc.fit(Y_train_mat)
Y_train_one_hot = enc.transform(Y_train_mat).toarray()

n_classes = Y_train_one_hot.shape[1]
n_features = X_train.shape[1]
n_train = X_train.shape[0]

def model_fn(X, Y_one_hot):
    with tf.variable_scope("logistic_regression"):
        weights = tf.get_variable('weights', [n_features, n_classes])
        bias = tf.get_variable('bias', [n_classes])
        logits = nn.xw_plus_b(X, weights, bias)
        y_probs = nn.softmax(logits)
        loss = loss_ops.softmax(logits, Y_one_hot)
        return y_probs, loss
 
g = tf.Graph()
# Data feeder converts to one-hot, and handles minibatching.
#data_feeder = setup_train_data_feeder(X_train, y_train, n_classes, batch_size)  
with g.as_default():    
    seed = 42
    random_seed.set_random_seed(seed)                                      
    #inp, out = data_feeder.input_builder() 
    inp = tf.placeholder(tf.float32, [None, n_features])
    out = tf.placeholder(tf.float32, [None, n_classes])
    global_step = tf.Variable(0, name="global_step", trainable=False)
    model_predictions, model_loss = model_fn(inp, out)   
    #train_op = tf.train.AdagradOptimizer(lr, clip_gradients=5).minimize(model_loss)   
    train_op = optimizers.optimize_loss(model_loss, global_step,
                learning_rate=lr, optimizer="Adagrad", clip_gradients=5)
    init_op = tf.initialize_all_variables()

sess = tf.Session(graph=g)

# Init
sess.run(init_op) 
    
# Train
    
train_loss_trace = []
#dict_feed_fn = data_feeder.get_feed_dict_fn()
for i in range(n_epochs+1):
    #feed_dict = dict_feed_fn() # {'input:0': X, 'output:0': Y }
    feed_dict = {inp: X_train, out: Y_train_one_hot}
    step, loss, _ = sess.run([global_step, model_loss, train_op], feed_dict=feed_dict)
    train_loss_trace.append(loss)
    if i % 50 == 0:
        print("iter {}, step {}, avg loss {}".format(i, step, np.mean(train_loss_trace)))

# Extract parameters
weights = sess.run("logistic_regression/weights:0")
bias = sess.run("logistic_regression/bias:0")
print('weights {}, bias {}'.format(weights, bias))

# Eval
feed_dict = {inp: X_train}
#feed_dict = {u'input:0': X_train}
Y_probs = sess.run(model_predictions, feed_dict=feed_dict)
y_pred_train = np.argmax(Y_probs, 1)

feed_dict = {inp: X_test}
#feed_dict = {u'input:0': X_test}
Y_probs = sess.run(model_predictions, feed_dict=feed_dict)
y_pred_test = np.argmax(Y_probs, 1)

sess.close()

score_train = accuracy_score(y_pred_train, y_train)
print('Train accuracy: %f' % score_train)
#Train accuracy: 0.950000


score_test = accuracy_score(y_pred_test, y_test)
print('Test accuracy: %f' % score_test)
#Test accuracy: 1.000000





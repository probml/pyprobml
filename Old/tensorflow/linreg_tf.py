# 1d linear regression in TF
#http://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import random_seed

seed = 42
random_seed.set_random_seed(seed)

# Define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)
# Plot input data
plt.scatter(X_data, y_data)

# Define data size and batch size
n_samples = 1000
batch_size = 100
X_data = np.reshape(X_data, (n_samples,1))
y_data = np.reshape(y_data, (n_samples,1))

# OLS solution
X1 = np.c_[X_data, np.ones(n_samples)]
w_ols = np.linalg.lstsq(X1, y_data)[0]
print 'OLS {}'.format(w_ols)

g = tf.Graph() 
with g.as_default():
    
    # Define placeholders for input
    X = tf.placeholder(tf.float32, shape=(None, 1))
    y = tf.placeholder(tf.float32, shape=(None, 1)) 
    
    # Define model
    with tf.variable_scope("linear-regression"):
        W = tf.get_variable("weights", (1, 1), initializer=tf.random_normal_initializer())
        b = tf.get_variable("bias", (1,), initializer=tf.constant_initializer(0.0))
        y_pred = tf.matmul(X, W) + b
        loss = tf.reduce_sum((y - y_pred)**2/n_samples)
            
    # Add train and initialize ops
    opt_operation = tf.train.AdamOptimizer(0.1).minimize(loss)
    init_op = tf.initialize_all_variables()

# Fit
sess = tf.Session(graph=g)

sess.run(init_op)
# Fit with SGD
n_epochs = 200
for _ in range(n_epochs):
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]
    _, loss_val = sess.run([opt_operation, loss], feed_dict={X: X_batch, y: y_batch})

weights = sess.run("linear-regression/weights:0")
bias = sess.run("linear-regression/bias:0")
print('weights {}, bias {}'.format(weights, bias))

# Eval            
y_pred = sess.run(y_pred, feed_dict={X: X_data})
sess.close()

plt.plot(X_data, y_pred, '-')
plt.show()


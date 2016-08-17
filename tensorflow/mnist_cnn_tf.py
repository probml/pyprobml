"""MNIST classifier using 2 layer convnet.
Based on https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.examples.tutorials.mnist import input_data


import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IM_SIZE = 28
IM_PIXELS = IM_SIZE * IM_SIZE


## Helper functions         
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  # The generated values follow a normal distribution with specified mean and standard deviation,
  #except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

## Build model
x = tf.placeholder(tf.float32, [None, IM_PIXELS])
x_image = tf.reshape(x, [-1,IM_SIZE,IM_SIZE,1])
y_true = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Conv layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    
h_pool1 = max_pool_2x2(h_conv1)

# Conv layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Compute h_pool2 size
hp2_size = 1
for d in h_pool2.get_shape()[1:].as_list():
    hp2_size *= d
# After two 2x2 maxpooling layers, image should be 4x smaller in each dimnension
assert hp2_size == (IM_SIZE // 4) * (IM_SIZE // 4) *64


# FC layer 1
W_fc1 = weight_variable([hp2_size, 1024]) 
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, hp2_size])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output layer
W_fc2 = weight_variable([1024, NUM_CLASSES])
b_fc2 = bias_variable([NUM_CLASSES])

y_pred_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_pred = tf.nn.softmax(y_pred_logits)

## Loss function and optimizer

#cross_entropy = -tf.reduce_sum(y_true*tf.log(y_conv))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_pred_logits, y_true)
# If y_true is not one-hot encoded, use this:
# tf.nn.sparse_softmax_cross_entropy_with_logits

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Print model shape

print("Shape of each layer")
print(x_image.get_shape())
print(h_conv1.get_shape())
print(h_pool1.get_shape())
print(h_conv2.get_shape())
print(h_pool2.get_shape())
print(h_pool2_flat.get_shape())
print(h_fc1.get_shape())


### Training

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
start_time = time.time()
sess = tf.InteractiveSession()
#with tf.Session() as sess:
tf.initialize_all_variables().run()
print('Initialized!')
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_true: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})
elapsed_time = time.time() - start_time
print("training wall clock time {}".format(elapsed_time))

## Testing
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images,
    y_true: mnist.test.labels, keep_prob: 1.0})
print("test accuracy {0:0.2f}".format(test_accuracy))

# After 1k steps, test accuracy should be 0.96 (153 seconds)
# After 20k steps, test accuracy should be 0.992 (half hour)

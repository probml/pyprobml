import tensorflow as tf
import numpy as np

# objective is f(x) = x1^2 + x2^3  
# derivative is [2 x1, 3 x2^2]

def cost_and_grad(y):
    session = tf.Session()
    #Y = tf.Variable(tf.placeholder(tf.float32, shape=[2]))
    Y = tf.placeholder(tf.float32, shape=[2])
    cost = tf.reduce_sum(tf.square(Y[0]) + tf.pow(Y[1], 3))
    cost_val =  session.run(cost, feed_dict = {Y: y})
    tfgrad = tf.gradients(cost, Y)
    grad_val = session.run(tfgrad, feed_dict = {Y: y})
    return cost_val, grad_val
    
# if x=[2,3], f(x) = 31, f'(x) = [4, 27]
f, g  =  cost_and_grad(np.array([2,3]))
print f
print g
assert(np.allclose(f, 31))
assert(np.allclose(g, np.array([4, 27])))

def cost_and_grad2(y):
    session = tf.Session()
    Y = tf.Variable(tf.placeholder(tf.float32, shape=[2]))
    cost = tf.reduce_sum(tf.square(Y[0]) + tf.pow(Y[1], 3))
    tfgrad = tf.gradients(cost, Y)[0]
    cost_val, grad_val = session.run([cost, tfgrad], feed_dict = {Y: y})
    return cost_val, grad_val
    
print cost_and_grad2(np.array([2,3]))

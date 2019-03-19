#https://github.com/pymanopt/pymanopt/blob/master/pymanopt/core/problem.py

from pymanopt import Problem

import tensorflow as tf
import numpy as np

X = tf.Variable(tf.placeholder(tf.float32, shape=[1]))
cost = tf.reduce_sum(tf.square(X))

problem = Problem(manifold=None, cost=cost, arg=X, verbosity=1)
print problem.cost(5)
print problem.egrad(5.0)


 
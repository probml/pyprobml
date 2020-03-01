# Nonlinear regression using variational inference for parameters.
# For simplicity we treat output noise variance as a fixed parameter.
# Adapted from
# https://brendanhasz.github.io/2019/07/23/bayesian-density-net.html


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
tf.keras.backend.set_floatx('float32')

import svi_mlp_regression_model_tfp as svimlp

import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from sklearn.metrics import mean_absolute_error, make_scorer


sns.reset_defaults()
#sns.set_style('whitegrid')
#sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

tfd = tfp.distributions


np.random.seed(12345)
tf.random.set_seed(12345)

## Make data

x_range = [-20, 60] # test
#x_ranges = [[-20, -10], [0, 20], [40, 50]]
#ns = [10, 10, 10]

#x_ranges = [ [-10,-5], [15,25], [35,50]]
#ns = [400, 400, 400]

x_ranges = [ [-20, 60]]
ns = [1000]

def load_dataset():
  def s(x): #std of noise
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return (0.25 + g**2.)
  x = []
  y = []
  for i in range(len(ns)):
    n = ns[i]
    xr = x_ranges[i]
    #x1 = (xr[1] - xr[0]) * np.random.rand(n) + xr[0]
    x1 = np.linspace(xr[0], xr[1], n)
    eps = np.random.randn(n) 
    #y1 = (1 * np.sin(0.2*x1) + 0.1 * x1) + eps*s(x1)
    y1 =  np.sin(0.2*x1) + eps
    x = np.concatenate((x, x1))
    y = np.concatenate((y, y1))
  print(x.shape)
  x = x[..., np.newaxis]
  #x = x.astype(np.float64)
  #y = y.astype(np.float64)
  n_tst = 150
  x_tst = np.linspace(*x_range, num=n_tst)
  #x_tst = x_tst.astype(np.float64)
  x_tst = x_tst[..., np.newaxis]
  return y, x, x_tst

y, x, x_tst = load_dataset()




plt.figure()
plt.plot(x, y, 'b.', label='observed');
plt.show()

# Make a TensorFlow Dataset from training data
BATCH_SIZE = 100
data_train = tf.data.Dataset.from_tensor_slices(
    (x, y)).shuffle(100).batch(BATCH_SIZE)

model1 = svimlp.BayesianDenseRegression([1, 50, 50, 1], flipout=False)

LR = 0.01
optimizer = tf.keras.optimizers.Adam(lr=LR)
N = x.shape[0]

@tf.function
def train_step(x_data, y_data):
    with tf.GradientTape() as tape:
        log_likelihoods = model1.log_likelihood(x_data, y_data)
        kl_loss = model1.losses
        elbo_loss = kl_loss/N - tf.reduce_mean(log_likelihoods)
    gradients = tape.gradient(elbo_loss, model1.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model1.trainable_variables))
    return elbo_loss

# Fit the model
EPOCHS = 100
elbo1 = np.zeros(EPOCHS)
for epoch in range(EPOCHS):
    for x_data, y_data in data_train:
        elbo1[epoch] += train_step(x_data, y_data)
   
# Plot the ELBO loss
plt.figure()
plt.plot(elbo1)
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.show()

plt.figure()
plt.plot(x, y, 'b.', label='observed');
pred = model1(x_tst, sampling=False)
m = pred[:,0]
s = pred[:,1]
plt.plot(x_tst, m, 'r', linewidth=4, label='mean')
plt.plot(x_tst, m + 2 * s, 'g', linewidth=2, label=r'mean + 2 stddev');
plt.plot(x_tst, m - 2 * s, 'g', linewidth=2, label=r'mean - 2 stddev');
plt.show()

    
# Nonlinear regression using variational inference for parameters.
# For simplicity we treat output noise variance as a fixed parameter.
# Adapted from
# https://brendanhasz.github.io/2019/07/23/bayesian-density-net.html


import superimport

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

#from sklearn.metrics import mean_absolute_error, make_scorer


sns.reset_defaults()
#sns.set_style('whitegrid')
#sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

tfd = tfp.distributions


np.random.seed(12345)
tf.random.set_seed(12345)

## Make data

x_range = [-20, 60] # test range (for plotting)

# choose intervals in which training data is observed
#x_ranges = [[-20, -10], [0, 20], [40, 50]]
#ns = [10, 10, 10]

x_ranges = [ [-20,-5], [5,25], [30,55]]
ns = [100, 100, 100]

#x_ranges = [ [-20, 60]]
#ns = [1000]

def load_dataset():
  def s(x): #std of noise
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return (0.25 + g**2.)
  x = []
  y = []
  for i in range(len(ns)):
    n = ns[i]
    xr = x_ranges[i] # range of observed data
    # x1 = locations within current range where we have data
    #x1 = (xr[1] - xr[0]) * np.random.rand(n) + xr[0]
    x1 = np.linspace(xr[0], xr[1], n)
    eps = np.random.randn(n)  # noise
    #y1 = (1 * np.sin(0.2*x1) + 0.1 * x1) + eps*s(x1)
    y1 =  np.sin(0.2*x1) + 0.1*eps
    x = np.concatenate((x, x1))
    y = np.concatenate((y, y1))
  print(x.shape)
  x = x[..., np.newaxis]
  x = x.astype(np.float32)
  y = y.astype(np.float32)
  n_tst = 150
  x_tst = np.linspace(*x_range, num=n_tst)
  x_tst = x_tst.astype(np.float32)
  x_tst = x_tst[..., np.newaxis]
  return y, x, x_tst

y, x, x_tst = load_dataset()
x_train = x
y_train= y[..., np.newaxis]
N = x_train.shape[0]


plt.figure()
plt.plot(x, y, 'b.', label='observed');
plt.show()

# Make a TensorFlow Dataset from training data
BATCH_SIZE = 100
data_train = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(N).batch(BATCH_SIZE)


# Fit the model

configs = [ 
          {'sampling': True, 'kl_factor': 1.0, 'kl_scaling': True, 'flipout': True},
          {'sampling': False, 'kl_factor': 0.0, 'kl_scaling': False, 'flipout': True},
        ]



nexpts = len(configs)
models = []
elbo_traces = []
for i in range(nexpts):
    ttl = 'experiment {}'.format(configs[i])
    print(ttl)
    sampling = configs[i]['sampling']
    use_kl_scaling = configs[i]['kl_scaling']
    kl_factor =  configs[i]['kl_factor']
    flipout = configs[i]['flipout']
    model = svimlp.BayesianDenseRegression([1, 50, 50, 1], flipout=flipout)

    LR = 0.01
    optimizer = tf.keras.optimizers.Adam(lr=LR)
    # this function relies on 'model' and 'optimizer' being in scope (yuk!)
    @tf.function(experimental_relax_shapes=True)
    def train_step(x_data, y_data, kl_multiplier=0.0, sampling=False):
        with tf.GradientTape() as tape:
            log_likelihoods = model.log_likelihood(x_data, y_data, sampling=sampling)
            kl_loss = model.losses
            elbo_loss = kl_multiplier*kl_loss/N - tf.reduce_mean(log_likelihoods)
        gradients = tape.gradient(elbo_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return elbo_loss

    EPOCHS = 1000
    elbo_trace = np.zeros(EPOCHS)
    if use_kl_scaling:
        kl_mults = kl_factor * np.logspace(0.0, 1.0, EPOCHS)/10.0
    else:
        kl_mults = kl_factor * np.ones(EPOCHS)
    for epoch in range(EPOCHS):
        for x_data, y_data in data_train:
            elbo_trace[epoch] += train_step(x_data, y_data, kl_mults[epoch], sampling)
    # Save trained model so we can plot stuff later
    models += [model]
    elbo_traces += [elbo_trace]
         
    # Plot the ELBO loss
    plt.figure()
    steps = range(10, EPOCHS)
    plt.plot(steps, elbo_trace[steps])
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.title(ttl)
    plt.show()
    
    plt.figure()
    plt.plot(x, y, 'b.', label='observed');
    pred = model(x_tst, sampling=False)
    m = pred[:,0]
    s = pred[:,1]
    plt.plot(x_tst, m, 'r', linewidth=4, label='mean')
    plt.plot(x_tst, m + 2 * s, 'g', linewidth=2, label=r'mean + 2 stddev');
    plt.plot(x_tst, m - 2 * s, 'g', linewidth=2, label=r'mean - 2 stddev');
    plt.title(ttl)
    plt.show()
    
        
    nsamples = 1000
    samples = model.samples(x_tst, nsamples) #ntst x nsamples  
    m = np.mean(samples, axis=1)
    s = np.std(samples, axis=1)
    n_tst = x_tst.shape[0]
    ndx = range(10, n_tst)
    plt.plot(x, y, 'b.', label='observed');
    plt.plot(x_tst[ndx], m[ndx], 'r', linewidth=4, label='mean')
    plt.plot(x_tst[ndx], m[ndx] + 2 * s[ndx], 'g', linewidth=2, label=r'mean + 2 stddev');
    plt.plot(x_tst[ndx], m[ndx] - 2 * s[ndx], 'g', linewidth=2, label=r'mean - 2 stddev');
    plt.title(ttl)
    plt.show()



for i in range(nexpts):
    model = models[i]
    elbo_trace = elbo_traces[i]
    ttl = 'experiment {}'.format(configs[i])
    
     # Plot the ELBO loss
    plt.figure()
    steps = range(10, EPOCHS)
    plt.plot(steps, elbo_trace[steps])
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.title(ttl)
    plt.show()
    
    plt.figure()
    plt.plot(x, y, 'b.', label='observed');
    pred = model(x_tst, sampling=False)
    m = pred[:,0]
    s = pred[:,1]
    plt.plot(x_tst, m, 'r', linewidth=4, label='mean')
    plt.plot(x_tst, m + 2 * s, 'g', linewidth=2, label=r'mean + 2 stddev');
    plt.plot(x_tst, m - 2 * s, 'g', linewidth=2, label=r'mean - 2 stddev');
    plt.title(ttl)
    plt.show()
    
        
    nsamples = 1000
    samples = model.samples(x_tst, nsamples) #ntst x nsamples  
    m = np.mean(samples, axis=1)
    s = np.std(samples, axis=1)
    n_tst = x_tst.shape[0]
    ndx = range(10, n_tst)
    plt.plot(x, y, 'b.', label='observed');
    plt.plot(x_tst[ndx], m[ndx], 'r', linewidth=4, label='mean')
    plt.plot(x_tst[ndx], m[ndx] + 2 * s[ndx], 'g', linewidth=2, label=r'mean + 2 stddev');
    plt.plot(x_tst[ndx], m[ndx] - 2 * s[ndx], 'g', linewidth=2, label=r'mean - 2 stddev');
    plt.title(ttl)
    plt.show()

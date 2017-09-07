
"""A multi-layer perceptron for classification of MNIST handwritten digits.
Based on 
#https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
"""

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.util import flatten
from autograd.optimizers import adam, rmsprop
from timeit import default_timer as timer

from examples.autograd import data 
#from data import load_mnist
print("Loading training data...")
N, train_images, train_labels, test_images,  test_labels = data.load_mnist()
 
#from examples import get_mnist
#(x_train, y_train, x_test, y_test) = get_mnist.get_mnist()
#(train_images, train_labels, test_images,  test_labels) =  get_mnist.get_mnist()


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def relu(x):       return np.maximum(0, x)
def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs) #np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


# Model parameters
#layer_sizes = [784, 200, 100, 10]
layer_sizes = [784, 512, 512, 10]
L2_reg = 1e-4  #1.0

# Training parameters
param_scale = 0.1
batch_size = 128 #256
num_epochs = 5
step_size = 0.001


init_params = init_random_params(param_scale, layer_sizes)

num_batches = int(np.ceil(len(train_images) / batch_size))
def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

# Define training objective
def objective(params, iter):
    idx = batch_indices(iter)
    return -log_posterior(params, train_images[idx], train_labels[idx], L2_reg)

# Get gradient of objective using autograd.
objective_grad = autograd.grad(objective)

print("     Epoch     |    Train accuracy  |       Test accuracy  ")
def print_perf(params, iter, gradient):
    if iter % num_batches == 0:
        train_acc = accuracy(params, train_images, train_labels)
        test_acc  = accuracy(params, test_images, test_labels)
        print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))

# The optimizers provided can optimize lists, tuples, or dicts of parameters.
start = timer()
optimizer = rmsprop # adam
optimized_params = optimizer(objective_grad, init_params, step_size=step_size,
                        num_iters=num_epochs * num_batches, callback=print_perf)
end = timer()
print('Training took {:f} seconds'.format(end - start))

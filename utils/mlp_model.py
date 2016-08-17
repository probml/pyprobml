# Multilayer perceptron for regression and classification
# Based on
# https://github.com/HIPS/autograd/blob/master/examples/neural_net_regression.py

import autograd
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd.scipy.misc import logsumexp

def relu(x):
    np.maximum(0, x)
      
class MLP(object):
    def __init__(self, layer_sizes, output_type='regression', L2_reg=0.001,
                nonlinearity=np.tanh):
        self.shapes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.nparams = sum((m+1)*n for m, n in self.shapes)
        self.L2_reg = L2_reg
        self.output_type = output_type
        self.nonlinearity = nonlinearity
        self.num_obj_fun_calls = 0
        self.num_grad_fun_calls = 0
        
    def init_params(self):
        param_scale = 0.1
        params = np.random.randn(self.nparams) * param_scale
        return params
        
    def unpack_layers(self, weights):
        for m, n in self.shapes:
            cur_layer_weights = weights[:m*n]     .reshape((m, n))
            cur_layer_biases  = weights[m*n:m*n+n].reshape((1, n))
            yield cur_layer_weights, cur_layer_biases
            weights = weights[(m+1)*n:]
            
    def predictions(self, W_vect, inputs):
        '''For classsification, returns N*C matrix of log probabilities.
        For rregression, returns N*K matrix of predicted means'''
        for W, b in self.unpack_layers(W_vect):
            outputs = np.dot(inputs, W) + b
            inputs = self.nonlinearity(outputs)
        if self.output_type == 'regression':
            return outputs
        if self.output_type == 'classification':
            logprobs = outputs - logsumexp(outputs, axis=1, keepdims=True)
            return logprobs
    
    def NLL(self, W_vect, X, Y, N=None):
        '''Negative log likelihood.
        For classification, we assume Y is a N*C one-hot matrix.'''
        if self.output_type == 'classification':
            log_lik = np.sum(self.predictions(W_vect, X) * Y)
        else:
            log_lik = 0
            Yhat = self.predictions(W_vect, X)
            Y = np.ravel(Y)
            Yhat = np.ravel(Yhat) 
            log_lik = -0.5*np.sum(np.square(Y - Yhat))
        if N is not None:
            # Compensate for this being a minibatch
            B = X.shape[0] # batch size
            log_lik = (log_lik / B ) * N
        return -log_lik 
        
    def PNLL(self, W_vect, X, Y, N=None):
        '''Penalized negative log likelihood.'''
        self.num_obj_fun_calls += 1
        log_prior = -self.L2_reg * np.dot(W_vect, W_vect)        
        log_lik = -self.NLL(W_vect, X, Y, N)
        return -(log_lik + log_prior)
            
    def gradient(self, W_vect, X, Y, N=None):
        self.num_grad_fun_calls += 1
        g = autograd.grad(self.PNLL)
        return g(W_vect, X, Y, N)

    
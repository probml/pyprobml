# Allow scipy.optimize.minimize to be used by Keras.
# https://github.com/fchollet/keras/issues/5085
# Original code by ncullen93
# Modified by murphyk so that get_updates has correct signature instead of
#   get_updates(self, params, constraints, loss)
#
# According to
# https://github.com/fchollet/keras/blob/master/keras/optimizers.py
# any keras optimizer has to implement the method
#    get_updates(self, loss, params)
# which should return a list of K.update(p, new_p) objects.


from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp

from keras import backend as K
from keras.optimizers import Optimizer


class ScipyOpt(Optimizer):
    """
    Scipy optimizer
    """
    def __init__(self, model, x, y, nb_epoch=500, method='L-BFGS-B', verbose=1, **kwargs):
        super(ScipyOpt, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.model = model
        self.x = x # input training data
        self.y = y # output training data
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.epoch_idx = 0
           

    def get_cost_grads(self, params, loss):
        """ 
        Get the loss and gradients of a Keras model.
        There are both TensorVariables.

        Arguments
        ---------
        params : list of trainable parameters (TensorVariables)

        loss : model loss function

        Returns
        -------
        loss : a TensorVariable
            The model loss

        grads : a list of TensorVariables
            Gradients of model params w.r.t. cost

        Effects
        -------
        None

        """
        grads = K.gradients(loss, params)
        return loss, grads

    def set_model_params(self, theta):
        """ 
        Sets the Keras model params from a flattened numpy array of theta 

        Arguments
        ---------
        theta : a flattened numpy ndarray
            The parameter values to set in the model

        Returns
        -------
        None

        Effects
        -------
        - Sets the model parameters to the values in theta

        """
        trainable_params = self.unpack_theta(theta)
        for trainable_param, layer in zip(trainable_params, self.model.layers):
            layer.set_weights(trainable_param)
            
    def unpack_theta(self, theta):
        """ 
        Converts flattened theta back to tensor shapes of Keras model params 
        
        Arguments
        ---------
        model : a compiled Keras model

        theta : a flattened numpy ndarray

        Returns
        -------
        weights : a list of numpy ndarrays in the shape of model params

        Effects
        -------
        None

        """        
        weights = []
        idx = 0
        for layer in self.model.layers:
            layer_weights = []
            for param in layer.get_weights():
                plen = np.prod(param.shape)
                layer_weights.append(np.asarray(theta[idx:(idx+plen)].reshape(param.shape),
                                        dtype=np.float32))
                idx += plen
            weights.append(layer_weights)
        return weights
    
    def pack_theta(self, trainable_weights):
        """ 
        Flattens a set of shared variables (trainable_weights)
        
        Arguments
        ---------
        trainable_weights : a list of shared variables

        Returns
        -------
        x : a flattened numpy ndarray of trainable weight values

        Effects
        -------
        None

        """        
        x = np.empty(0)
        for t in trainable_weights:
            x = np.concatenate((x,K.get_value(t).reshape(-1)))
        return x
    
    def flatten_grads(self, grads):
        """ 
        Flattens a set of TensorVariables

        Arguments
        ---------
        grads : a list of TensorVariables
            Gradients of model params

        Returns
        -------
        x : a flattened list of TensorVariables

        Effects
        -------
        None

        """       
        x = np.empty(0) 
        for g in grads:
            x = np.concatenate((x,g.reshape(-1)))
        return x
    
    def get_train_fn(self, params, loss):
        """
        Get Scipy training function that returns loss and gradients

        Arguments
        ---------
        params : a list of trainable keras TensorVariables

        loss : a TensorVariable

        Returns
        -------
        train_fn : a callable python function
            A scipy.optimize-compatible function returning loss & grads

        Effects
        -------
        - Alters the trainable parameters of the input Keras model here.
        """
        cost, grads = self.get_cost_grads(params, loss)
        outs = [cost]
        if type(grads) in {list, tuple}:
            outs += grads
        else:
            outs.append(grads) 
        if self.verbose > 0:
            print('Compiling Training Function..')
            
        # fn = K.function(inputs=[], outputs=outs,
        #         givens={self.model.model.inputs[0]          : self.x,
        #                 self.model.model.targets[0]         : self.y,
        #                 self.model.model.sample_weights[0]  : np.ones((self.x.shape[0],), dtype=np.float32),
    #                 K.learning_phase()                  : np.uint8(1)})
    # The above code has a 'givens' kwarg which is not supported by
    # https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py#L2277
        
                
        def train_fn(theta):
            self.set_model_params(theta)
            cost_grads = fn([])
            cost = np.asarray(cost_grads[0], dtype=np.float64)
            
            if self.verbose > 0:
                if self.epoch_idx % 1 == 0:
                    try:
                        print('Epoch : %i/%i Loss : %f' % (self.epoch_idx, 
                            self.nb_epoch, cost))
                    except ValueError:
                        pass
            grads = np.asarray(self.flatten_grads(cost_grads[1:]), dtype=np.float64)
            if self.verbose > 1:
                if self.epoch_idx % 25 == 0:
                    try:
                        print('Avg. Grad: ' , grads.mean())
                    except ValueError:
                        pass
            self.epoch_idx+=1
            return cost, grads

        return train_fn

    #def get_updates(self, params, constraints, loss):
    def get_updates(self, params, loss):  # murphyk
        #self.x = self.model.model.ins[0]
        #self.y = self.model.model.ins[1]
        #_params = params.copy()
        theta0 = self.pack_theta(params)
        train_fn = self.get_train_fn(params, loss)
        
        sp.optimize.minimize(train_fn, theta0,
            method=self.method, jac=True, 
            options={'maxiter':self.nb_epoch,'disp':False})

        #theta_final = weights.x
        self.updates = []
        #final_params = self.unpack_theta(theta_final)
        #for orig, final in zip(params, final_params):
         #   self.updates.append((orig, final))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'epsilon': self.epsilon}
        base_config = super(ScipyOpt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


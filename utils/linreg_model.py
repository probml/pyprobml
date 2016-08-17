'''Linear regression class'''

import autograd.numpy as np  # Thinly-wrapped numpy

def squared_loss(y_pred, y):
    N = y.shape[0]
    return 0.5*np.sum(np.square(y - y_pred))/N
    #return np.sum(np.square(y - y_pred))

class LinregModel(object):
    def __init__(self, ninputs, add_ones=False):
        if add_ones:
            ninputs = ninputs + 1
        self.ninputs = ninputs
        self.add_ones = add_ones
    
    def init_params(self):
        return np.zeros(self.ninputs)
        
    def maybe_add_column_of_ones(self, X):
        if self.add_ones:
            N = X.shape[0]
            X = np.c_[X, np.ones(N)]
        return X
            
    def prediction(self, params, X):
        X1 = self.maybe_add_column_of_ones(X)
        yhat = np.dot(X1, params)
        return yhat
    
    def objective(self, params, X, y):
        ypred = self.prediction(params, X)
        return squared_loss(ypred, y)
        
    def gradient(self, params, X, y):
        # gradient of objective = (1/N) sum_n x(n,:)*yerr(n)   // row vector
        y_pred = self.prediction(params, X)
        N = y.shape[0]
        yerr = np.reshape((y_pred - y), (N, 1))
        X1 = self.maybe_add_column_of_ones(X)
        gradient = np.sum(X1 * yerr, 0)/N # broadcast yerr along columns
        return gradient
  
    def ols_fit(self, X, y):
        X1 = self.maybe_add_column_of_ones(X)
        w_ols = np.linalg.lstsq(X1, y)[0]
        loss_ols = squared_loss(self.prediction(w_ols, X), y)
        return w_ols, loss_ols
 
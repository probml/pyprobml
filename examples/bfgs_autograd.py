from __future__ import print_function
import autograd
import autograd.numpy as np  # Thinly-wrapped numpy
from scipy.optimize import minimize

def squared_loss(y_pred, y):
    N = y.shape[0]
    return 0.5*sum(np.square(y - y_pred))/N

def bfgs_auto(Xtrain, ytrain):
    D = Xtrain.shape[1]
    params = np.zeros(D)
    obj_fun = lambda params: squared_loss(np.dot(Xtrain, params),  ytrain)
    grad_fun = autograd.grad(obj_fun)
    result = minimize(obj_fun, params,  method='BFGS', jac=grad_fun)
    return result.x, result.fun

def main():
    np.random.seed(1)
    N = 21
    D = 2
    Xtrain = np.random.randn(N, D)
    ytrain = np.random.randn(N)
    params, obj = bfgs_auto(Xtrain, ytrain)
    print(params)
    print(obj)

if __name__ == "__main__":
    main()

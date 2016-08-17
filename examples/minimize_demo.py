import autograd
import autograd.numpy as np
from scipy.optimize import minimize

count = 0
def fake(x):
    global count
    print count
    count += 1
    return x ** 4 + 10 * x ** 3 + 4 * x ** 2 + 7 * x + 1

obj_fun = fake
grad_fun = autograd.grad(obj_fun)
params = -1
num_iters = 20

result = minimize(obj_fun, params, method='BFGS', jac=grad_fun,
            options = {'maxiter':num_iters, 'disp':True})
            
# each gradient computation calls the function (because of autograd)
assert(result.nfev + result.njev == count)
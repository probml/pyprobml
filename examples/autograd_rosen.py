import autograd.numpy as np
from autograd import grad
from autograd.convenience_wrappers import hessian_vector_product as hvp
from scipy.optimize import minimize


def rosen(x):
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

result = minimize(
    rosen, x0, method='Newton-CG',
    jac=grad(rosen), hessp=hvp(rosen),
    options={'xtol': 1e-8, 'disp': True})
print result

result = minimize(
    rosen, x0, method='BFGS',
    jac=grad(rosen),
    options={'xtol': 1e-8, 'disp': True})
print result
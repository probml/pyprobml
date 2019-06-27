# illustrate automatic differentiation using jax
# https://github.com/google/jax
import numpy as onp # original numpy
import jax.numpy as np
from jax import grad

onp.random.seed(42)
D = 5
w = onp.random.randn(D) # jax handles RNG differently

x = onp.random.randn(D)
y = 0 # should be 0 or 1

def sigmoid(x): return 0.5 * (np.tanh(x / 2.) + 1)

#d/da sigmoid(a) = s(a) * (1-s(a))
deriv_sigmoid = lambda a: sigmoid(a) * (1-sigmoid(a))
deriv_sigmoid_jax = grad(sigmoid)
a0 = 1.5
assert np.isclose(deriv_sigmoid(a0), deriv_sigmoid_jax(a0))

# mu(w)=s(w'x), d/dw mu(w) = mu * (1-mu) .* x
def mu(w): return sigmoid(np.dot(w,x))
def deriv_mu(w): return mu(w) * (1-mu(w)) * x
deriv_mu_jax =  grad(mu)
assert np.allclose(deriv_mu(w), deriv_mu_jax(w))

# NLL(w) = -[y*log(mu) + (1-y)*log(1-mu)]
# d/dw NLL(w) = (mu-y)*x
def nll(w): return -(y*np.log(mu(w)) + (1-y)*np.log(1-mu(w)))
#def deriv_nll(w): return -(y*(1-mu(w))*x - (1-y)*mu(w)*x)
def deriv_nll(w): return (mu(w)-y)*x
deriv_nll_jax = grad(nll)
assert np.allclose(deriv_nll(w), deriv_nll_jax(w))


# Now do it for a batch of data


def predict(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

def loss(weights, inputs, targets):
    preds = predict(weights, inputs)
    logprobs = np.log(preds) * targets + np.log(1 - preds) * (1 - targets)
    return -np.sum(logprobs)

             
N = 3
X = onp.random.randn(N, D)
y = onp.random.randint(0, 2, N)

from jax import vmap
from functools import partial

preds = vmap(partial(predict, w))(X)  
preds2 = vmap(predict, in_axes=(None, 0))(w, X)
preds3 = [predict(w, x) for x in X]
preds4 = predict(w, X)
assert np.allclose(preds, preds2)
assert np.allclose(preds, preds3)
assert np.allclose(preds, preds4)dr

grad_fun = grad(loss)
grads = vmap(partial(grad_fun, w))(X,y)
assert grads.shape == (N,D)
grads2 = np.dot(np.diag(preds-y), X)
assert np.allclose(grads, grads2)

grad_sum = np.sum(grads, axis=0)
grad_sum2 = np.dot(np.ones((1,N)), grads)
assert np.allclose(grad_sum, grad_sum2)

# Now make things go fast
from jax import jit

grad_fun = jit(grad(loss))
grads = vmap(partial(grad_fun, w))(X,y)
assert np.allclose(grads, grads2)


# Now work with Jacobians

from jax import jacfwd, jacrev

Din = 3; Dout = 4;
A = onp.random.randn(Dout, Din)
def fun(x):
    return np.dot(A, x)
x = onp.random.randn(Din)
Jf = jacfwd(fun)(x)
Jr = jacrev(fun)(x)
assert np.allclose(Jf, Jr)
assert np.allclose(Jf, A)

# If the function outputs a scalar, the Jacobian is the gradient vector
Din = 3; Dout = 1;
A = onp.random.randn(Dout, Din)
def fun(x):
    return np.dot(A, x)[0]
x = onp.random.randn(Din)
J = jacrev(fun)(x)
g = grad(fun)(x)
assert np.allclose(J, g)

 
def obj(w):
    return loss(w, X, y)
grad0 = lambda w: grad(loss)(w, X, y)
grad1 = grad(obj)
grad2 = jacrev(obj)
assert np.allclose(grad1(w), grad2(w))

# Now work with Hessians

def hessian(fun):
  return jacfwd(jacrev(fun))

# Quadratic form
A = onp.random.randn(D, D)
x = onp.random.randn(D)
myfun = lambda x: np.dot(x, np.dot(A, x))
H = hessian(myfun)(x)
assert np.allclose(H, A+A.T)

# Logistic regression
H1 = hessian(loss)(w, X, y)
mu = predict(w, X)
S = np.diag(mu * (1-mu))
H2 = np.dot(np.dot(X.T, S), X)
assert np.allclose(H1, H2)

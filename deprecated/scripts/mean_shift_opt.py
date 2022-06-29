# mean shift to find global modes
import superimport

import numpy as  np # original numpy
#import jax.numpy as jnp
#from jax import vmap
import numpy as np
from functools import partial
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname)) #os.path.join(figdir, fname)

def kernelfn_scalar(x, lam=0.4, beta=10):
    if np.abs(x) > lam: 
        return 0.0
    else:
        return np.exp(-beta*x**2)
    
#def kernel_broken(xs):
#    kernels = vmap(kernelfn_scalar)(xs)
#    return kernels

kernelfn =  np.vectorize(kernelfn_scalar)

def objfn(xs):
    weights = [0.5, 0.5]
    mu = [-0.5, 0.5]
    sigmas = [0.2, 0.1]
    dist0 = norm(loc=mu[0], scale=sigmas[0])
    dist1 = norm(loc=mu[1], scale=sigmas[1])
    return weights[0]*dist0.pdf(xs) + weights[1]*dist1.pdf(xs)

def weightfn_scalar(S, s):
    fn =  np.vectorize(lambda t: kernelfn_scalar(t-s))
    vals = fn(S)
    denom =  np.sum(vals)
    return objfn(s) / denom

def weightfn(S, xs):
    #fn  =  np.vectorize(partial(weightfn_scalar, S))
    fn = objfn
    return fn(xs)

def qfn_scalar(S, x):
    fn =  np.vectorize(lambda s: kernelfn_scalar(x-s) * weightfn_scalar(S,s))
    vals = fn(S)
    return  np.sum(vals)

def qfn(S, xs):
    fn  =  np.vectorize(partial(qfn_scalar, S))
    return fn(xs)
    
def meanshift_scalar(S, x):
    fn =  np.vectorize(lambda s: kernelfn_scalar(s-x) * weightfn_scalar(S,s) * s)
    numer = fn(S)
    fn =  np.vectorize(lambda s: kernelfn_scalar(s-x) * weightfn_scalar(S,s))
    denom = fn(S)
    return np.sum(numer) / np.sum(denom+1e-10)

def meanshift(S, xs):
    fn  =  np.vectorize(partial(meanshift_scalar, S))
    return fn(xs)


grid = np.linspace(-1, 1, num=50)
plt.figure()
plt.plot(grid, objfn(grid))
save_fig('meanshift-target.pdf')

np.random.seed(42)
dist = uniform(loc=-1, scale=2) #-1..1
S = dist.rvs(size=20)
#S = grid 


for i in range(4):
    plt.figure()
    plt.stem(S, objfn(S))
    plt.xlim([-1, 1])
    plt.title('S{}'.format(i))
    save_fig('meanshift-samples-iter{}.pdf'.format(i))
    
    plt.figure()
    q = qfn(S, grid)
    plt.plot(grid, q)
    plt.title('q{}'.format(i))
    save_fig('meanshift-density-iter{}.pdf'.format(i))
    
    plt.show()
    
    S = meanshift(S,S)

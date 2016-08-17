#Modfified from http://pymc-devs.github.io/pymc3/getting_started/

import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal, find_MAP, NUTS, sample,Slice, traceplot, summary
import scipy.optimize as opt

np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

size = 100

X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

basic_model = Model()
with basic_model:
    alpha = Normal('alpha', mu=0, sd=10)
    beta = Normal('beta', mu=0, sd=10, shape=2)
    sigma = HalfNormal('sigma', sd=1)
    mu = alpha + beta[0]*X1 + beta[1]*X2
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
    
print basic_model
    
map_estimate = find_MAP(model=basic_model)
print(map_estimate)
#{'alpha': array(0.9065985664354854), 'beta': array([ 0.948486  ,  2.60705513]), 'sigma_log': array(-0.03278146854842092)}

map_estimate2 = find_MAP(model=basic_model, fmin=opt.fmin_powell)
print(map_estimate2)

map_estimate3 = find_MAP(model=basic_model, fmin=opt.fmin_l_bfgs_b)
print(map_estimate3)

with basic_model:
    #step = Slice(vars=[sigma]) 
    #trace = sample(2000, start=map_estimate, step=step) 
    trace = sample(2000, start=map_estimate) 

beta_samples = trace['beta'] # shape 2000, 2
traceplot(trace);
summary(trace)
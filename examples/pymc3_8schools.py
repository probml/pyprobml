# Modified from
# https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/gelman_schools.py

import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal, HalfCauchy, sample, traceplot, loo


J = 8
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])


# Schools model defined at https://raw.githubusercontent.com/wiki/stan-dev/rstan/8schools.stan
with Model() as schools:
    print 'building model...'
    eta = Normal('eta', 0, 1, shape=J)
    mu = Normal('mu', 0, sd=1e6)
    tau = HalfCauchy('tau', 25) # original model uses U[0,infty]
    theta = mu + tau*eta
    obs = Normal('obs', theta, sd=sigma, observed=y)
    
    
with schools:
    print 'sampling...'
    tr = sample(1000)
    l = loo(tr) # -29.6821436703
    print 'LOO estimate {}'.format(l)
    
traceplot(tr)


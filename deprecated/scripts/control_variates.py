# Control variates demo
#https://en.wikipedia.org/wiki/Control_variates

import superimport

import numpy as np
np.random.seed(0)

N = 1500
u = np.random.uniform(size=N)
f = 1 / (1+u)
mu_naive = np.mean(f)
se_naive  = np.sqrt(np.var(f)/N)
print('naive {:0.4f}, se {:0.4f}'.format(mu_naive, se_naive))

# control variate version
c = 0.4773
g = 1+u 
baseline = 3.0/2
cv = f + c*(g - baseline)
mu_cv = np.mean(cv)
se_cv  = np.sqrt(np.var(cv)/N)
print('cv {:0.4f}, se {:0.4f}'.format(mu_cv, se_cv))



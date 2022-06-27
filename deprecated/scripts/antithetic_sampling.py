# demo of antithetic sampling
# https://en.wikipedia.org/wiki/Antithetic_variates


import superimport

import numpy as np
np.random.seed(0)

N = 750 #1500
u1 = np.random.uniform(size=N)
u2 = np.random.uniform(size=N)
u = np.concatenate((u1,u2))
f = 1 / (1+u)
mu_naive = np.mean(f)
se_naive  = np.sqrt(np.var(f)/(2*N))
print('naive {:0.4f}, se {:0.4f}'.format(mu_naive, se_naive))

# antithetic version
uprime = 1-u1
f1 = 1 / (1+u1)
fprime = 1 / (1+uprime)
f = (f1 + fprime) / 2.0 # paired samples!
mu_anti = np.mean(f)
se_anti  = np.sqrt(np.var(f)/(2*N))
print('anti {:0.4f}, se {:0.4f}'.format(mu_anti, se_anti))
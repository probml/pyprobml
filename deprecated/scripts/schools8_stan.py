#https://pystan.readthedocs.org/en/latest/getting_started.html
import superimport

import pystan
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

schools_code = """
data {
    int<lower=0> J; // number of schools
    real y[J]; // estimated treatment effects
    real<lower=0> sigma[J]; // s.e. of effect estimates
}
parameters {
    real mu;
    real<lower=0> tau;
    real eta[J];
}
transformed parameters {
    real theta[J];
    for (j in 1:J)
        theta[j] <- mu + tau * eta[j];
}
model {
    # priot on tau is implicitly U[0,infty]
    eta ~ normal(0, 1);
    #y ~ normal(theta, sigma)
    for (j in 1:J)
       y[j] ~ normal(theta[j], sigma[j]);
}
"""

# The prior for tau is implicitly ...

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}


# Since compiling the model is slow, we want to do it once and save it
# as a pickle file. We follow the suggestions from
# https://pystan.readthedocs.org/en/latest/avoiding_recompilation.html
fname = 'stan_8schools_model.pkl'
if os.path.isfile(fname):
    print('loading {}'.format(fname))
    sm = pickle.load(open(fname, 'rb'))
else:
    print('compiling model...')
    sm = pystan.StanModel(model_code=schools_code)
    with open(fname, 'wb') as f:
        print('saving {}'.format(fname))
        pickle.dump(sm, f)

# Run MCMC
nchains = 2
fit = sm.sampling(data=schools_dat, iter=1000, chains=nchains)

samples = fit.extract(permuted=True)  # dictionary, remove warmup
eta_samples = samples['eta']
eta_samples.shape # 1000,8

fit.plot()

plt.show()

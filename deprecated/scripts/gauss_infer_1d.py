# Implementation of gaussInferParamsMean1d
# Author: Animesh Gupta

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pyprobml_utils as pml

priorVar = [1, 5]
Sigma = 1 # Assumed to be known
X = 3 
post_mu = 0
post_Sigma = 0

for i in priorVar:
    
    prior_Sigma = i
    prior_mu    = 0
    xbar = np.mean(X)
    n    = np.size(X)
    
    lik_Sigma = Sigma
    lik_mu    = xbar
    
    S0    = prior_Sigma
    S0inv = 1./S0
    mu0   = prior_mu
    S     = Sigma
    Sinv  = 1./S
    Sn    = 1./(S0inv + n*Sinv)
    
    post_mu    = Sn*(n*Sinv*xbar + S0inv*mu0)
    post_Sigma = Sn
    
    x = np.arange(-5,5.25, 0.25)

    fig = plt.figure(figsize = (4, 4))
    plt.ylim(0,0.6)
    plt.xlim(-5,5)
    plt.xticks([-5,0,5])
    plt.plot(x, multivariate_normal.pdf(x, mean = prior_mu, cov = prior_Sigma), color = "blue", 
             label = "prior")
    plt.plot(x, multivariate_normal.pdf(x, mean = lik_mu, cov = lik_Sigma), color = "red", 
             label = "lik",linestyle='dotted')
    plt.plot(x, multivariate_normal.pdf(x, mean = post_mu, cov = post_Sigma), color = "black", 
             label = "post",linestyle='dashdot')
    plt.title(f"prior variance of {i}")
    plt.legend(loc="upper left")
    pml.savefig(f"gauss_infer_1d_{i}.pdf")

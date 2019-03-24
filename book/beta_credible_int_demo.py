# Compute 95% CI for a beta distribution
from scipy.stats import beta
import numpy as np
np.random.seed(42)

N = 100; N1 = 47; N0 = N-N1 # Sufficient statistics of likelihood
aprior = 1; bprior = 1; # prior
apost = aprior + N1; bpost = bprior + N0 # posterior

alpha = 0.05
CI1 = beta.interval(1-alpha, apost, bpost)
print(CI1) # (0.3749, 0.567)

l  = beta.ppf(alpha/2, apost, bpost)
u  = beta.ppf(1-alpha/2, apost, bpost)
CI2 = (l,u)
print(CI2) # (0.3749, 0.567)

samples = beta.rvs(apost, bpost, size=1000)
samples = np.sort(samples)
CI3 = np.percentile(samples, 100*np.array([alpha/2, 1-alpha/2])) 
print(CI3) # (0.372, 0.564)

from scipy.stats import beta
import numpy as np

S = 47
N = 100 
a = S+1
b = (N-S)+1 
alpha = 0.05;

CI1 = beta.interval(1-alpha, a, b)

l  = beta.ppf(alpha/2, a, b)
u  = beta.ppf(1-alpha/2, a, b)
CI2 = (l,u)

samples = beta.rvs(a, b, size=1000)
samples = np.sort(samples)
CI3 = np.percentile(samples, 100*np.array([alpha/2, 1-alpha/2])) 

print CI1
print CI2
print CI3

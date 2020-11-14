# Example 5.1 from "Test and roll: profit maximizing A/B tests"
# https://pubsonline.informs.org/doi/abs/10.1287/mksc.2019.1194


import numpy as np

def optimal_sample_size(N, s, sigma):
    # eqn 10
    t = (s/sigma)**2
    n = np.sqrt(0.25*N*t + (0.75*t)**2) - 0.75*t
    return n

def prob_error(n1, n2, s, sigma):
    # eqn 12
    x = np.sqrt(2)*sigma/s * np.sqrt(n1*n2/(n1+n2))
    p = 0.25 - 1/(2*np.pi)*np.arctan(x)
    return 2*p # could have m1<m2 or m1>m2

def eprofit(N, n1, n2, s, sigma):
    # eqn 9
    numer = np.sqrt(2)*sigma**2
    tmp = 2*sigma**2 + (n1+n2) / (n1*n2) * (s**2)
    denom = np.sqrt(np.pi)*np.sqrt(tmp)
    return (N-n1-n2)*(mu + numer/denom)

mu = 0.68
sigma = 0.03
N = 100000
s = np.sqrt(mu*(1-mu))
n = optimal_sample_size(N, s, sigma)
print(n) # 2283.9

n1 = n
n2 = n
p = prob_error(n1, n2, s, sigma)
print(p) # 0.10

Eprofit_test = (n1+n2)*mu
Eprofit_deploy = eprofit(N, n1, n2, s, sigma)
print(Eprofit_test) # 3016.0
print(Eprofit_deploy) # 66429.9
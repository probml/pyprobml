import superimport

import numpy as np
from pyprobml_utils import save_fig
import matplotlib.pyplot as plt
from scipy.special import betaln

theta = 0.7
N = 5
alpha = 1
alphaH = alpha
alphaT = alpha

# instantiate a number of datastructures
flips = np.zeros((2**N, N))
Nh = np.zeros(2**N)
Nt = np.zeros(2**N)
marginal_lik = np.zeros(2**N)
log_lik = np.zeros(2**N)
log_BF = np.zeros(2**N)

for i in range(0,2**N):
    flips[i] = np.array(np.unravel_index(i, [2]*N, 'F')) + 1
    Nh[i] = len(np.where(flips[i] == 1)[0])
    Nt[i] = len(np.where(flips[i] == 2)[0])
    marginal_lik[i] = np.exp(betaln(alphaH+Nh[i], alphaT+Nt[i]) - betaln(alphaH, alphaT))
    mle = Nh[i] / N
    log_lik[i] = Nh[i]*np.log10(mle + 10e-8) + Nt[i]*np.log10(1 - mle + 10e-8)
    log_BF[i] = betaln(alphaH+Nh[i], alphaT+Nt[i]) - betaln(alphaH, alphaH) - N*np.log(0.5)

#sort in order of number of heads
ndx = np.argsort(Nh)
Nh = Nh[ndx]
marginal_lik = marginal_lik[ndx]
log_lik = log_lik[ndx]
log_BF = log_BF[ndx]

p0 = (1/2)**N
plt.plot(marginal_lik, 'o-', linewidth=2)
plt.plot((0,2**N), (p0,p0), c='k', linewidth=2)
plt.xticks(list(range(len(Nh))), Nh.astype(int))
plt.xlabel('num heads')
plt.title(r"Marginal likelihood for Beta-Bernoulli model $\int p(D|\theta) Be(\theta | 1, 1,) d\Theta$")
plt.xlim((-0.6,2**N))
save_fig("joshCoins4.pdf")
plt.show()

plt.plot(np.exp(log_BF), 'o-', linewidth=2)
plt.title("BF(1,0)")
plt.xticks(list(range(len(Nh))), Nh.astype(int))
plt.xlim((-0.6,2**N))
save_fig("joshCoins4BF.pdf")
plt.show()


BIC1 = log_lik - 1
plt.plot(BIC1, 'o-', linewidth=2)
plt.title(r"BIC approximation to $log_{10} p(D|M1)$")
plt.xticks(list(range(len(Nh))), Nh.astype(int))
plt.xlim((-0.6,2**N))
save_fig("joshCoins4BIC.pdf")
plt.show()

plt.plot(np.log10(marginal_lik), 'o-', linewidth=2)
plt.title(r"$log_{10} p(D | M1)$")
plt.xticks(list(range(len(Nh))), Nh.astype(int))
plt.xlim((-0.6,2**N))
save_fig("joshCoins4LML.pdf")
plt.show()

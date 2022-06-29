import superimport

import pyprobml_utils as pml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy
#import scipy.special as sc
from scipy import stats
from sklearn.neighbors import KernelDensity
import math

S = np.array([[3.1653, -0.0262], [-0.0262, 0.6477]])
dof = 3
Sigma = S
nr = 3
nc = 3
nsamples = 9
Xs1 = np.linspace(0.1, 200, 2000)
Xs2 = np.linspace(0.1, 40, 400)
Xs3 = np.linspace(0.1, 10, 100)
Xsa = {0:Xs1, 1:Xs2} 
Xsb = {0:Xs2, 1:Xs3}

def wishart_sample(dof, sigma, nsamples):
  d = np.size(sigma, 0)
  C = np.linalg.cholesky(Sigma)
  S = np.zeros((d, d, nsamples))
  for i in range(nsamples):
      t = np.random.randn(dof, d) 
      Z = np.matmul(t, C)
      S[:, :, i] = np.matmul(Z.T, Z)
  return S

def gamma_log_prob(a, b, X):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    # a=shape, b=rate=1/scale
  logZ = scipy.special.gammaln(a) - np.multiply(a, np.log(b))
  logp = np.multiply((a-1), np.log(X)) - np.multiply(b, X) - logZ
  return logp

def get_cov_ellipse(cov, centre, nstd): 
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)
    theta = math.degrees(theta)
    width, height = 2 * nstd * np.sqrt(eigvals)

    return width, height, theta

np.random.seed(4) 
M = dof*Sigma
R = np.corrcoef(M)
S = wishart_sample(dof, Sigma, nsamples)

# Plots of some samples from Wishart distribution:
fig, ax = plt.subplots(nr, nc, figsize=(10,10))
mu = np.array([0, 0])
j = 0  
for r in range(nr):
    for c in range(nc):
      w, h, theta = get_cov_ellipse(S[:, :, j], mu, nstd=3)
      j = j + 1
      ell = mpl.patches.Ellipse(xy=mu, width=w, height=h, angle = theta, ec='black', fc='none') #facecolor='none'
      ax[r, c].add_patch(ell)
      ax[r, c].set_aspect('equal')
      ax[r, c].set_xlim(-25, 25)
      ax[r, c].set_ylim(-25, 25)
      ax[r, c].autoscale()
      ax[r, c].plot(mu[0], mu[1], marker='x')
plt.tight_layout()
fig.suptitle("Wi(dof=%i, S), E=[9.5, -0.1; -0.1, 1.9], rho=-0.018" %dof)
xmin = -17
xmax = 17
ymin = -10
ymax = 10
custom_xlim = (xmin, xmax)
custom_ylim = (ymin, ymax)
plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
pml.savefig('wishart_samples.pdf')
plt.show()

marg1a = dof / 2
marg1b = 1/(2*Sigma[0, 0])
marg2a = dof / 2
marg2b = 1/(2*Sigma[1, 1])

#Plots of marginals 
logp = gamma_log_prob(marg1a, marg1b, Xs1)
expo = np.exp(logp)
plt.figure()
plt.plot(Xs1, expo)
plt.title("Marginal sigma1_squared ")
pml.savefig('wishart_sigma1.pdf')
plt.show()

logp = gamma_log_prob(marg2a, marg2b, Xs2)
expo = np.exp(logp)
plt.figure()
plt.plot(Xs2, expo)
plt.title("Marginal sigma2_squared ")
pml.savefig('wishart_sigma2.pdf')
plt.show()

# Plot of correlation coefficient
n = 1000
Rs = wishart_sample(dof, Sigma, nsamples)
for s in range(nsamples):
        Rs[:, :, s] = np.corrcoef(Rs[:, :, s])    #cov2cor(Rs(:, :, s));
data = np.squeeze(Rs[0, 1, :])
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(data.reshape(-1, 1)) 
x = np.linspace(data.min()-2, data.max()+2, 100)
logprob = kde.score_samples(x[:, None])
plt.figure()
plt.title('Rho')
plt.plot(x, np.exp(logprob))
pml.savefig('wishart_rho.pdf')
plt.show()

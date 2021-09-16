# Online learning of a logistic
# regression model using the Exponential-family
# Extended Kalman Filter (EEKF) algorithm

# Author: Gerardo Durán-Martín (@gerdm)

import superimport

import jax
import nlds_lib as ds
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from jax import random
from jax.scipy.optimize import minimize
from sklearn.datasets import make_biclusters

def sigmoid(x): return jnp.exp(x) / (1 + jnp.exp(x))
def log_sigmoid(z): return z - jnp.log(1 + jnp.exp(z))
def fz(x): return x
def fx(w, x): return sigmoid(w[None, :] @ x)
def Rt(w, x): return sigmoid(w @ x) * (1 - sigmoid(w @ x))

## Data generating process
n_datapoints = 50
m = 2
X, rows, cols = make_biclusters((n_datapoints, m), 2,
                                noise=0.6, random_state=314,
                                minval=-4, maxval=4)
# whether datapoints belong to class 1
y = rows[0] * 1.0

Phi = jnp.c_[jnp.ones(n_datapoints)[:, None], X]
N, M = Phi.shape

colors = ["black" if el else "white" for el in y]

# Predictive domain
xmin, ymin = X.min(axis=0) - 0.1
xmax, ymax = X.max(axis=0) + 0.1
step = 0.1
Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
_, nx, ny = Xspace.shape
Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

### EEKF Approximation
mu_t = jnp.zeros(M)
Pt = jnp.eye(M) * 0.0
P0 = jnp.eye(M) * 2.0

model = ds.ExtendedKalmanFilter(fz, fx, Pt, Rt)
w_eekf_hist, P_eekf_hist = model.filter(mu_t, y, Phi, P0)

w_eekf = w_eekf_hist[-1]
P_eekf = P_eekf_hist[-1]

### Laplace approximation
key = random.PRNGKey(314)
init_noise = 0.6
w0 = random.multivariate_normal(key, jnp.zeros(M), jnp.eye(M) * init_noise)
alpha = 1.0
def E(w):
    an = Phi @ w
    log_an = log_sigmoid(an)
    log_likelihood_term = y * log_an + (1 - y) * jnp.log1p(-sigmoid(an))
    prior_term = alpha * w @ w / 2

    return prior_term - log_likelihood_term.sum()

res = minimize(lambda x: E(x) / len(y), w0, method="BFGS")
w_laplace = res.x
SN = jax.hessian(E)(w_laplace)

### Ploting surface predictive distribution
key = random.PRNGKey(31415)
nsamples = 5000

# EEKF surface predictive distribution
eekf_samples = random.multivariate_normal(key, w_eekf, P_eekf, (nsamples,))
Z_eekf = sigmoid(jnp.einsum("mij,sm->sij", Phispace, eekf_samples))
Z_eekf = Z_eekf.mean(axis=0)

fig, ax = plt.subplots()
ax.contourf(*Xspace, Z_eekf, cmap="RdBu_r", alpha=0.7, levels=20)
ax.scatter(*X.T, c=colors, edgecolors="black", s=80)
ax.set_title("(EEKF) Predictive distribution")
pml.savefig("logistic-regression-surface-eekf.pdf")

# Laplace surface predictive distribution
laplace_samples = random.multivariate_normal(key, w_laplace, SN, (nsamples,))
Z_laplace = sigmoid(jnp.einsum("mij,sm->sij", Phispace, laplace_samples))
Z_laplace = Z_laplace.mean(axis=0)

fig, ax = plt.subplots()
ax.contourf(*Xspace, Z_laplace, cmap="RdBu_r", alpha=0.7, levels=20)
ax.scatter(*X.T, c=colors, edgecolors="black", s=80)
ax.set_title("(Laplace) Predictive distribution")
pml.savefig("logistic-regression-surface-laplace.pdf")

### Plot EEKF and Laplace training history
P_eekf_hist_diag = jnp.diagonal(P_eekf_hist, axis1=1, axis2=2)
P_laplace_diag = jnp.sqrt(jnp.diagonal(SN))
lcolors = ["black", "tab:blue", "tab:red"]
elements = w_eekf_hist.T, P_eekf_hist_diag.T, w_laplace, P_laplace_diag, lcolors
timesteps = jnp.arange(n_datapoints) + 1

for k, (wk, Pk, wk_laplace, Pk_laplace, c) in enumerate(zip(*elements)):
    fig, ax = plt.subplots()
    ax.errorbar(timesteps, wk, jnp.sqrt(Pk), c=c, label=f"$w_{k}$ online (EEKF)")
    ax.axhline(y=wk_laplace, c=c, linestyle="dotted", label=f"$w_{k}$ batch (Laplace)", linewidth=3)

    ax.set_xlim(1, n_datapoints)
    ax.legend(framealpha=0.7, loc="upper right")
    ax.set_xlabel("number samples")
    ax.set_ylabel("weights")
    plt.tight_layout()
    pml.savefig(f"eekf-laplace-hist-w{k}.pdf")

print("EEKF weights")
print(w_eekf, end="\n"*2)

print("Laplace weights")
print(w_laplace, end="\n"*2)
plt.show()


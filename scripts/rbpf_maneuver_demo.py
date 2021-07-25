# Rao-Blackwellised particle filtering for jump markov linear systems
# Based on: https://github.com/probml/pmtk3/blob/master/demos/rbpfManeuverDemo.m
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install matplotlib==3.4.2

import jax
import numpy as np
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
import particle_filtering_lib as pflib
import pyprobml_utils as pml
from jax import random
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from jax.scipy.special import logit
from numpy import linalg

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


def kdeg(x, X, h):
    """
    KDE under a gaussian kernel
    
    Parameters
    ----------
    x: array(eval, D)
    X: array(obs, D)
    h: float
    Returns
    -------
    array(eval):
        KDE around the observed values
    """
    N, D = X.shape
    nden, _ = x.shape 
    
    Xhat = X.reshape(D, 1, N)
    xhat = x.reshape(D, nden, 1)
    u = xhat - Xhat
    u = linalg.norm(u, ord=2, axis=0) ** 2 / (2 * h ** 2)
    px = np.exp(-u).sum(axis=1) / (N * h * np.sqrt(2 * np.pi))
    return px


def scale_3d(ax, x_scale, y_scale, z_scale, factor):
    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3,3] = factor
    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)
    return short_proj


def style3d(ax, x_scale, y_scale, z_scale, factor=0.62):
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.get_proj = scale_3d(ax, x_scale, y_scale, z_scale, factor=factor)


TT = 0.1
A = jnp.array([[1, TT, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, TT],
               [0, 0, 0, 1]])


B1 = jnp.array([0, 0, 0, 0])
B2 = jnp.array([-1.225, -0.35, 1.225, 0.35])
B3 = jnp.array([1.225, 0.35,  -1.225,  -0.35])
B = jnp.stack([B1, B2, B3], axis=0)

Q = 0.2 * jnp.eye(4)
R = 10 * jnp.diag(jnp.array([2, 1, 2, 1]))
C = jnp.eye(4)

transition_matrix = jnp.array([
    [0.9, 0.05, 0.05],
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9]
])

transition_matrix = jnp.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])

params = pflib.RBPFParamsDiscrete(A, B, C, Q, R, transition_matrix)

nparticles = 1000
nsteps = 100
key = random.PRNGKey(1)
keys = random.split(key, nsteps)

x0 = (1, random.multivariate_normal(key, jnp.zeros(4), jnp.eye(4)))
draw_state_fixed = partial(pflib.draw_state, params=params)

# Create target dataset
_, (latent_hist, state_hist, obs_hist) = jax.lax.scan(draw_state_fixed, x0, keys)

# Perform filtering
key_base = random.PRNGKey(31)
key_mean_init, key_sample, key_state, key_next = random.split(key_base, 4)
p_init = jnp.array([0.0, 1.0, 0.0])

# Initial filter configuration
mu_0 = 0.01 * random.normal(key_mean_init, (nparticles, 4))
mu_0 = 0.01 * random.normal(key_mean_init, (nparticles, 4))
Sigma_0 = jnp.zeros((nparticles, 4,4))
s0 = random.categorical(key_state, logit(p_init), shape=(nparticles,))
weights_0 = jnp.ones(nparticles) / nparticles
init_config = (key_next, mu_0, Sigma_0, weights_0, s0)

rbpf_optimal_part = partial(pflib.rbpf_optimal, params=params, nparticles=nparticles)
_, (mu_hist, Sigma_hist, weights_hist, s_hist, Ptk) = jax.lax.scan(rbpf_optimal_part, init_config, obs_hist)
mu_hist_post_mean = jnp.einsum("ts,tsm->tm", weights_hist, mu_hist)


# Plot target dataset
color_dict = {0: "tab:green", 1: "tab:red", 2: "tab:blue"}
fig, ax = plt.subplots()
color_states_org = [color_dict[state] for state in latent_hist]
ax.scatter(*state_hist[:, [0, 2]].T, c="none", edgecolors=color_states_org, s=10)
ax.scatter(*obs_hist[:, [0, 2]].T, s=5, c="black", alpha=0.6)
pml.savefig("rbpf-maneuver-data.pdf")

# Plot filtered dataset
fig, ax = plt.subplots()
rbpf_mse = ((mu_hist_post_mean - state_hist)[:, [0, 2]] ** 2).mean(axis=0).sum()
latent_hist_est = Ptk.mean(axis=1).argmax(axis=1)
color_states_est = [color_dict[state] for state in latent_hist_est]
ax.scatter(*mu_hist_post_mean[:, [0, 2]].T, c="none", edgecolors=color_states_est, s=10)
ax.set_title(f"RBPF MSE: {rbpf_mse:.2f}")
pml.savefig("rbpf-maneuver-trace.pdf")

# Plot belief state of discrete system
p_terms = Ptk.mean(axis=1)
rbpf_error_rate = (latent_hist != p_terms.argmax(axis=1)).mean()
fig, ax = plt.subplots(figsize=(2.5, 5))
sns.heatmap(p_terms, cmap="viridis", cbar=False)
plt.title(f"RBPF, error rate: {rbpf_error_rate:0.3}")
pml.savefig("rbpf-maneuver-discrete-belief.pdf")

# Plot ground truth and MAP estimate
ohe = OneHotEncoder(sparse=False)
latent_hmap = ohe.fit_transform(latent_hist[:, None])
latent_hmap_est = ohe.fit_transform(p_terms.argmax(axis=1)[:, None])

fig, ax = plt.subplots(figsize=(2.5, 5))
sns.heatmap(latent_hmap, cmap="viridis", cbar=False, ax=ax)
ax.set_title("Data")
pml.savefig("rbpf-maneuver-discrete-ground-truth.pdf")

fig, ax = plt.subplots(figsize=(2.5, 5))
sns.heatmap(latent_hmap, cmap="viridis", cbar=False, ax=ax)
ax.set_title(f"MAP (error rate: {rbpf_error_rate:0.4f})")
pml.savefig("rbpf-maneuver-discrete-map.pdf")

# Plot belief for state space
skip = 3
dim = 2
npoints = 2000
azimuth, elevation = -30, 30
xmin, xmax = mu_hist[..., dim].min(), mu_hist[..., dim].max()
xrange = jnp.linspace(xmin, xmax, npoints).reshape(-1, 1)
res = np.apply_along_axis(lambda X: kdeg(xrange, X[..., None], 0.5), 1, mu_hist)
densities = res[..., dim]

fig = plt.figure()
axs = plt.axes(projection="3d")
for t in range(0, nsteps, skip):
    tloc = t * np.ones(npoints)
    px = densities[t]
    axs.plot(tloc, xrange, px, c="tab:blue", linewidth=1)
axs.set_zlim(0, 1)
style3d(axs, 1.8, 1.2, 0.7, 0.8)
axs.view_init(elevation, azimuth)
axs.set_xlabel(r"$t$", fontsize=13)
axs.set_ylabel(r"$x_{d,t}$", fontsize=13)
axs.set_zlabel(r"$p(x_{d, t} \vert y_{1:t})$", fontsize=13)
pml.savefig("rbpf-maneuver-belief-state.pdf", pad_inches=0, bbox_inches="tight")

plt.show()

import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
import particle_filtering_lib as pflib
from jax import random
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from jax.scipy.special import logit

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

color_dict = {0: "tab:green", 1: "tab:red", 2: "tab:blue"}

# Plot target dataset
fig, ax = plt.subplots()
color_states_org = [color_dict[state] for state in latent_hist]
ax.scatter(*state_hist[:, [0, 2]].T, c="none", edgecolors=color_states_org, s=10)
ax.scatter(*obs_hist[:, [0, 2]].T, s=5, c="black", alpha=0.6)

# Plot filtered dataset
fig, ax = plt.subplots()
rbpf_mse = ((mu_hist_post_mean - state_hist)[:, [0, 2]] ** 2).mean(axis=0).sum()
latent_hist_est = Ptk.mean(axis=1).argmax(axis=1)
color_states_est = [color_dict[state] for state in latent_hist_est]
ax.scatter(*mu_hist_post_mean[:, [0, 2]].T, c="none", edgecolors=color_states_est, s=10)
ax.set_title(f"RBPF MSE: {rbpf_mse:.2f}")

p_terms = Ptk.mean(axis=1)
rbpf_error_rate = (latent_hist != p_terms.argmax(axis=1)).mean()
fig, ax = plt.subplots(figsize=(2.5, 5))
sns.heatmap(p_terms, cmap="viridis", cbar=False)
plt.title(f"RBPF, error rate: {rbpf_error_rate:0.3}")

ohe = OneHotEncoder(sparse=False)
latent_hmap = ohe.fit_transform(latent_hist[:, None])
latent_hmap_est = ohe.fit_transform(p_terms.argmax(axis=1)[:, None])

fig, ax = plt.subplots(1, 2, figsize=(4 + 1, 5))
sns.heatmap(latent_hmap, cmap="viridis", cbar=False, ax=ax[0])
sns.heatmap(latent_hmap_est, cmap="viridis", cbar=False, ax=ax[1])
ax[0].set_title("Data")
ax[1].set_title(f"MAP (error rate: {rbpf_error_rate:0.4f})")

plt.show()
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from jax.scipy.special import logit
from dataclasses import dataclass

import matplotlib.pyplot as plt

@dataclass
class RBPFParamsDiscrete:
    """
    Rao-Blackwell Particle Filtering (RBPF) parameters for
    a system with discrete latent-space.
    We assume that the system evolves as
        z_next = A * z_old + B(u_old) + noise1_next
        x_next = C * z_next + noise2_next
        u_next ~ transition_matrix(u_old)
    
    where
        noise1_next ~ N(0, Q)
        noise2_next ~ N(0, R)
    """
    A: jnp.array
    B: jnp.array
    C: jnp.array
    Q: jnp.array
    R: jnp.array
    transition_matrix: jnp.array


def draw_state(val, key, params):
    """
    Simulate one step of a system that evolves as
                A z_{t-1} + Bk + eps,
    where eps ~ N(0, Q).
    
    Parameters
    ----------
    val: tuple (int, jnp.array)
        (latent value of system, state value of system).
    params: PRBPFParamsDiscrete
    key: PRNGKey
    """
    latent_old, state_old = val
    probabilities = params.transition_matrix[latent_old, :]
    logits = logit(probabilities)
    latent_new = random.categorical(key, logits)
    
    key_latent, key_obs = random.split(key)
    state_new = params.A @ state_old + params.B[latent_new, :]
    state_new = random.multivariate_normal(key_latent, state_new, params.Q)
    obs_new = random.multivariate_normal(key_obs, params.C @ state_new, params.R)
    
    return (latent_new, state_new), (latent_new, state_new, obs_new)


def kf_update(mu_t, Sigma_t, k, xt, params):
    I = jnp.eye(len(mu_t))
    mu_t_cond = params.A @ mu_t + params.B[k]
    Sigma_t_cond = params.A @ Sigma_t @ params.A.T + params.Q
    xt_cond = params.C @ mu_t_cond
    St = params.C @ Sigma_t_cond @ params.C.T + params.R
    
    Kt = Sigma_t_cond @ params.C.T @ jnp.linalg.inv(St)
    
    # Estimation update
    mu_t = mu_t_cond + Kt @ (xt - xt_cond)
    Sigma_t  = (I - Kt @ params.C) @ Sigma_t_cond
    
    # Normalisation constant
    mean_norm = params.C @ mu_t_cond
    cov_norm = params.C @ Sigma_t_cond @ params.C.T + params.R
    Ltk = jax.scipy.stats.multivariate_normal.pdf(xt, mean_norm, cov_norm)
    
    return mu_t, Sigma_t, Ltk


def rbpf_step(key, weight_t, st, mu_t, Sigma_t, xt, params):
    log_p_next = logit(params.transition_matrix[st])
    k = random.categorical(key, log_p_next)
    mu_t, Sigma_t, Ltk = kf_update(mu_t, Sigma_t, k, xt, params)
    weight_t = weight_t * Ltk
    
    return mu_t, Sigma_t, weight_t, Ltk


kf_update_vmap = jax.vmap(kf_update, in_axes=(None, None, 0, None, None), out_axes=0)


def rbpf_step_optimal(key, weight_t, st, mu_t, Sigma_t, xt, params):    
    k = jnp.arange(len(params.transition_matrix))
    mu_tk, Sigma_tk, Ltk = kf_update_vmap(mu_t, Sigma_t, k, xt, params)
    
    proposal = Ltk * transition_matrix[st]
    
    weight_tk = weight_t * proposal.sum()
    proposal = proposal / proposal.sum()
    
    return mu_tk, Sigma_tk, weight_tk, proposal


# vectorised RBPF step
rbpf_step_vec = jax.vmap(rbpf_step, in_axes=(0, 0, 0, 0, 0, None, None))
# vectorisedRBPF Step optimal
rbpf_step_optimal_vec = jax.vmap(rbpf_step_optimal, in_axes=(0, 0, 0, 0, 0, None, None))


def rbpf(current_config, xt, params, nparticles=100):
    """
    Rao-Blackwell Particle Filter using prior as proposal
    """
    key, mu_t, Sigma_t, weights_t, st = current_config
    
    key_sample, key_state, key_next, key_reindex = random.split(key, 4)
    keys = random.split(key_sample, nparticles)
    
    st = random.categorical(key_state, logit(params.transition_matrix[st, :]))
    mu_t, Sigma_t, weights_t, Ltk = rbpf_step_vec(keys, weights_t, st, mu_t, Sigma_t, xt, params)
    weights_t = weights_t / weights_t.sum()
    
    indices = jnp.arange(nparticles)
    pi = random.choice(key_reindex, indices, shape=(nparticles,), p=weights_t, replace=True)
#     pi = random.categorical(key_reindex, logit(weights_t), shape=(nparticles, ))
    st = st[pi]
    mu_t = mu_t[pi, ...]
    Sigma_t = Sigma_t[pi, ...]
    weights_t = jnp.ones(nparticles) / nparticles
    
    return (key_next, mu_t, Sigma_t, weights_t, st), (mu_t, Sigma_t, weights_t, st, Ltk)


def rbpf_optimal(current_config, xt, params, nparticles=100):
    """
    Rao-Blackwell Particle Filter using optimal proposal
    """
    key, mu_t, Sigma_t, weights_t, st = current_config
    
    key_sample, key_state, key_next, key_reindex = random.split(key, 4)
    keys = random.split(key_sample, nparticles)
    
    st = random.categorical(key_state, logit(params.transition_matrix[st, :]))
    mu_t, Sigma_t, weights_t, proposal = rbpf_step_optimal_vec(keys, weights_t, st, mu_t, Sigma_t, xt, params)
    
    indices = jnp.arange(nparticles)
    pi = random.choice(key_reindex, indices, shape=(nparticles,), p=weights_t, replace=True)
    
    # Obtain optimal proposal distribution
    proposal_samp = proposal[pi, :]
    st = random.categorical(key, logit(proposal_samp))
    
    mu_t = mu_t[pi, st, ...]
    Sigma_t = Sigma_t[pi, st, ...]
    
    weights_t = jnp.ones(nparticles) / nparticles
    
    return (key_next, mu_t, Sigma_t, weights_t, st), (mu_t, Sigma_t, weights_t, st, proposal_samp)


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
R = 3 * jnp.diag(jnp.array([2, 1, 2, 1]))
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

params = RBPFParamsDiscrete(A, B, C, Q, R, transition_matrix)

nparticles = 1000
nsteps = 100
key = random.PRNGKey(1)
keys = random.split(key, nsteps)

x0 = (1, random.multivariate_normal(key, jnp.zeros(4), jnp.eye(4)))
draw_state_fixed = partial(draw_state, params=params)

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

rbpf_optimal_part = partial(rbpf_optimal, params=params, nparticles=nparticles)
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

plt.show()

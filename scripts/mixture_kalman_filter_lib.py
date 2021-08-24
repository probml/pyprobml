# Mixture Kalman Filter library. Also known as the
# Rao-Blackwell Particle Filter.

# Author: Gerardo Durán-Martín (@gerdm)

import superimport

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logit
from dataclasses import dataclass

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
    
    proposal = Ltk * params.transition_matrix[st]
    
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

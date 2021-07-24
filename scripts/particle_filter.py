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


TT = 0.5
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
key = random.PRNGKey(3141)
keys = random.split(key, nsteps)


x0 = (1, random.multivariate_normal(key, jnp.zeros(4), jnp.eye(4)))
draw_state_fixed = partial(draw_state, params=params)

_, (latent_hist, state_hist, obs_hist) = jax.lax.scan(draw_state_fixed, x0, keys)

color_dict = {0: "tab:green", 1: "tab:red", 2: "tab:blue"}
color_states_org = [color_dict[state] for state in latent_hist]
plt.scatter(*state_hist[:, [0, 2]].T, c="none", edgecolors=color_states_org, s=10)
plt.scatter(*obs_hist[:, [0, 2]].T, s=5, c="black", alpha=0.6)
plt.show()

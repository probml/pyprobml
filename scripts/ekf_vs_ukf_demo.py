# Compare extended Kalman filter with unscented kalman filter on a nonlinear 2d tracking problem
# Author: Gerardo Durán-Martín (@gerdm)

import nlds_lib as ds
import matplotlib.pyplot as plt
import pyprobml_utils as pml
import jax.numpy as jnp
from jax import random

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

def check_symmetric(a, rtol=1.1):
    return jnp.allclose(a, a.T, rtol=rtol)


def fz(x, dt):
    return x + dt * jnp.array([jnp.sin(x[1]), jnp.cos(x[0])])
def fx(x): return x


dt = 0.4
nsteps = 100

# Initial state vector
x0 = jnp.array([1.5, 0.0])

# State noise
Qt = jnp.eye(2) * 0.001
# Observed noise
Rt = jnp.eye(2) * 0.05
alpha, beta, kappa = 1, 0, 2

key = random.PRNGKey(314)
model = ds.ExtendedKalmanFilter(lambda x: fz(x, dt), fx, Qt, Rt)
sample_state, sample_obs = model.sample(key, x0, nsteps)


def plot_data(sample_state, sample_obs):
    fig, ax = plt.subplots()
    ax.plot(*sample_state.T, label="state space")
    ax.scatter(*sample_obs.T, s=60, c="tab:green", marker="+")
    ax.scatter(*sample_state[0], c="black", zorder=3)
    ax.legend()
    ax.set_title("Noisy observations from hidden trajectory")
    plt.axis("equal")


def plot_inference(sample_obs, mean_hist, Sigma_hist):
    fig, ax = plt.subplots()
    ax.scatter(*sample_obs.T, marker="+", color="tab:green")
    ax.plot(*mean_hist.T, c="tab:orange", label="filtered")
    ax.scatter(*mean_hist[0], c="black", zorder=3)
    plt.legend()
    collection = [(mut, Vt) for mut, Vt in zip(mean_hist[::4], Sigma_hist[::4])
                  if Vt[0, 0] > 0 and Vt[1, 1] > 0 and abs(Vt[1, 0] - Vt[0, 1]) < 7e-4]
    for mut, Vt in collection:
        pml.plot_ellipse(Vt, mut, ax, plot_center=False, alpha=0.9, zorder=3)
    plt.axis("equal")


plot_data(sample_state, sample_obs)
pml.savefig("nlds2d_data.pdf")

ekf = ds.ExtendedKalmanFilter(lambda x: fz(x, dt), fx, Qt, Rt)
mean_hist, Sigma_hist = ekf.filter(x0, sample_obs)
plot_inference(sample_obs, mean_hist, Sigma_hist)
plt.title('EKF')
pml.savefig("nlds2d_ekf.pdf")

alpha, beta, kappa = 1, 0, 2
ukf = ds.UnscentedKalmanFilter(lambda x: fz(x, dt), fx, Qt, Rt, alpha, beta, kappa)
mean_hist, Sigma_hist = ukf.filter(x0, sample_obs)
plot_inference(sample_obs, mean_hist, Sigma_hist)
plt.title('UKF')
pml.savefig("nlds2d_ukf.pdf")

plt.show()

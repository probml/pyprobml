# Example showcasing the learning process of the EKF algorithm.
# This demo is based on the ekf_mlp_anim_demo.py demo.
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

import jax.numpy as jnp
from jsl.demos import ekf_mlp_anim_demo

filepath = "./../figures/ekf_mlp_demo.mp4"
def fx(x): return x -10 * jnp.cos(x) * jnp.sin(x) + x ** 3
def fz(W): return W

ekf_mlp_anim_demo.main(fx, fz, filepath)

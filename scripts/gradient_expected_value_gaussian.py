# Bonnet's thm: d/dm E[f(z)] = E[d/dz f(z)] for z ~ N(m,v)
#Price's thm:  d/dv E[f(z)] = 0.5 E[d^2/dz^2 f(z)] for z ~ N(m,v)

# Note that we are taking derivatives wrt the parameters of the sampling distribution
# We rely on the fact that TFP Gaussian samples are reparameterizable

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
tfp = tfp.substrates.jax
tfd = tfp.distributions

key = jax.random.PRNGKey(0)
nsamples = 10000


def f(z):
  return jnp.square(z) # arbitrary fn

def expect_f(params):
  m, v = params
  dist = tfd.Normal(m, jnp.sqrt(v))
  zs = dist.sample(nsamples, key)
  return jnp.mean(f(zs))

def expect_grad(params):
  m, v = params
  dist = tfd.Normal(m, jnp.sqrt(v))
  zs = dist.sample(nsamples, key)
  grads = jax.vmap(jax.grad(f))(zs)
  return jnp.mean(grads)

def expect_grad2(params):
  m, v = params
  dist = tfd.Normal(m, jnp.sqrt(v))
  zs = dist.sample(nsamples, key)
  #g = jax.grad(f)
  #grads = jax.vmap(jax.grad(g))(zs)
  grads = jax.vmap(jax.hessian(f))(zs)
  return jnp.mean(grads)


params = (1.0, 2.0)


e1 = expect_grad(params)
e2 = expect_grad2(params)
print([e1, 0.5*e2])

grads = jax.grad(expect_f)(params)
print(grads)

assert np.allclose(e1, grads[0], atol=1e-1)
assert np.allclose(0.5 * e2, grads[1], atol=1e-1)


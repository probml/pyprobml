import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
tfp = tfp.substrates.jax
tfd = tfp.distributions

key = jax.random.PRNGKey(0)
nsamples = 10

# Bonnet's thm: d/dm E[f(z)] = E[d/dz f(z)] for z ~ N(m,v)
# Note that we are taking derivatives wrt the parameters of the sampling distribution
# We rely on the fact that TFP Gaussian samples are reparameterizable

def f(z):
  return jnp.square(z) # arbitrary fn

def expect_f(params):
  m, v = params
  dist = tfd.Normal(m, jnp.sqrt(v))
  zs = dist.sample(nsamples, key)
  return jnp.mean(f(zs))

def expect_grad(params, order=1):
  m, v = params
  dist = tfd.Normal(m, jnp.sqrt(v))
  zs = dist.sample(nsamples, key)
  grads = jax.vmap(jax.grad(f))(zs)
  return jnp.mean(grads)


params = (1.0, 2.0)

print(expect_grad(params, ))

grads = jax.grad(expect_f)(params)
print(grads[0])
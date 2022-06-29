'''
NAGVAC is an alternative to cholesky decomposition and it uses the factor decomposition which is
    \Sigma = BB^T + C^2
where B is the factor loading matrix of size (d,f) for d << f the number of factors and C is a diagonal matrix.
For more information, please see https://vbayeslab.github.io/VBLabDocs/tutorial/ffvb/vafc.
This implementation is the Python version of https://github.com/VBayesLab/VBLab/blob/main/VBLab/VB/NAGVAC.m .
Author: Aleyna Kara(@karalleyna)
'''

import jax.numpy as jnp
from jax.random import normal, split
from jax import lax, tree_map, vmap, value_and_grad

import optax

from vb_utils import clip


def compute_natural_gradients(b, c, grads):
    '''
    Computes natural gradients which is equivalent to (I^-1 x grad).

    Parameters
    ----------
    b : Array
        The vector factor loading vector component of the variational covariance matrix

    c : Array
        The diagonal matrix component of the variational covariance matrix

    grad : Tuple
        It is triple containing the gradients of mean, b, c respectively.

    Returns
    -------
    Tuple : Includes inverse fisher information times gradient for each gradient given.
    '''
    b_square, c_square = b ** 2, c ** 2

    grad_mu, grad_b, grad_c = grads

    v_1 = c_square - 2 * b_square * (1 / c_square) ** 2
    v_2 = b_square - (1 / c_square * c)

    kappa_1 = jnp.sum(b_square / c_square)
    kappa_2 = (1 / (1 + jnp.sum(v_2 ** 2 / v_1))) / 2.

    nat_grad_mu = (grad_mu.T @ b) * b + c_square * grad_mu

    coef = (1 + kappa_1) / kappa_2
    nat_grad_b = coef * (grad_b.T @ b) * b + c_square * grad_b

    nat_grad_c = (grad_c / v_1) / 2.
    tmp = (v_2 / v_1)
    nat_grad_c += kappa_2 * (tmp.T @ grad_c) * tmp

    return nat_grad_mu, nat_grad_b, nat_grad_c


def grad_log_q_function(b, c, theta, mu):
    x = theta - mu
    d = b / c ** 2
    grad_log_q = -x / c ** 2 + (d.T @ x) / (1 + (d.T @ b)) * d
    return grad_log_q


def vb_gauss_lowrank(key, logjoint_fn, data, nfeatures,
                     initial_mean=None, initial_std=0.1,
                     initial_scale=1., nsamples=20,
                     niters=200, optimizer=optax.adafactor(1e-3),
                     threshold=2500, window_size=None):
    '''
    Parameters
    ----------
    key : jax.random.PRNGKey

    logjoint_fn : Callable
        Log joint function

    data : Tuple
        The data to which the model is fitted, specified as a table or matrix.

    nfeatures :
        Number of features

    initial_mean :

    initial_std :
        Standard deviation of normal distribution for initialization

    initial_scale : float
        The constant factor  to scale the initial values.

    num_samples : int
        Monte Carlo samples to estimate the lower bound

    niters : int
        Maximum number of iterations

    optimizer : optax.optimizers

    threshold : float
        Gradient clipping threshold

    window_size : int
        Rolling window size to smooth the lower bound.
        Default value of window size is None,  which indicates that lower bounds won't be smoothed.

    Returns
    -------
    Tuple: Consists of
            1. mu : Estimation of variational mean
            2. b : The vector factor loading vector component of the variational covariance matrix
            3. c : The diagonal matrix component of the variational covariance matrix

    Array : Estimation of the lower bound over iterations

    '''
    if initial_mean is None:
        mu_key, key = split(key, 2)
        mu = initial_std * normal(mu_key, shape=(nfeatures, 1))
    else:
        mu = initial_mean

    b_key, key = split(key, 2)
    b = initial_std * normal(b_key, shape=(nfeatures, 1))
    c = initial_scale * jnp.ones((nfeatures, 1))

    # Variational parameters vector
    variational_params = (mu, b, c)

    # Initial state of the optimizer
    opt_state = optimizer.init(variational_params)

    def sample_fn(variational_params, U_normal):
        mu, b, c = variational_params

        # Parameters in Normal distribution
        epsilon1 = U_normal[0]
        epsilon2 = U_normal[1:].reshape((-1, 1))

        theta = mu + b * epsilon1 + c * epsilon2
        h_theta, grad_h_theta = value_and_grad(logjoint_fn)(theta, data)

        # Gradient of log variational distribution
        grad_log_q = grad_log_q_function(b, c, theta, mu)

        # Gradient of h(theta) and lower bound
        grad_theta = grad_h_theta - grad_log_q

        return grad_theta, epsilon1 * grad_theta, epsilon2 * grad_theta, h_theta

    def iter_fn(all_params, key):
        # Main VB iteration

        variational_params, opt_state = all_params
        mu, b, c = variational_params
        samples = normal(key, shape=(nsamples, nfeatures + 1))

        *grad_lb_iter, lb_first_term = vmap(sample_fn, in_axes=(None, 0))(variational_params, samples)

        # Estimation of lowerbound
        logdet = jnp.log(jnp.linalg.det(1 + (b / c ** 2).T @ b)) + jnp.sum(jnp.log(c ** 2))

        # Mean of log-q -> mean(log q(theta))
        lb_log_q = -0.5 * nfeatures * jnp.log(2 * jnp.pi) - 0.5 * logdet - nfeatures / 2
        lower_bound = jnp.mean(lb_first_term) - lb_log_q

        # Gradient of log variational distribution
        grad_lb = tree_map(lambda x: x.mean(axis=0), grad_lb_iter)
        grads = compute_natural_gradients(b, c, grad_lb)

        # Gradient clipping
        grads = clip(grads, threshold=threshold)

        updates, opt_state = optimizer.update(grads, opt_state, variational_params)
        variational_params = optax.apply_updates(variational_params, updates)
        return (variational_params, opt_state), (variational_params, lower_bound)

    keys = split(key, niters)
    (best_params, _), (variational_params, lower_bounds) = lax.scan(iter_fn, (variational_params, opt_state), keys)

    if window_size is not None:
        def simple_moving_average(cur_sum, i):
            diff = (lower_bounds[i] - lower_bounds[i - window_size]) / window_size
            cur_sum += diff
            return cur_sum, cur_sum

        indices = jnp.arange(window_size, niters)
        cur_sum = jnp.sum(lower_bounds[:window_size]) / window_size
        _, lower_bounds = lax.scan(simple_moving_average, cur_sum, indices)
        lower_bounds = jnp.append(jnp.array([cur_sum]), lower_bounds)

        i = jnp.argmax(lower_bounds) + window_size - 1
        best_params = tree_map(lambda x: x[i], variational_params)

    return best_params, lower_bounds

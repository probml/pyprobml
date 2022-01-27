import jax.numpy as jnp
from jax.random import normal, split
from jax import lax, tree_map, value_and_grad, tree_leaves

import optax


# I^-1 x grad
def inverse_fisher_times_grad(b, c, grad):
    d = b.size
    grad1, grad2, grad3 = grad
    c2 = c ** 2
    b2 = b ** 2

    prod1 = (b.T @ grad1) * b + (grad1 * c2)
    alpha = 1 / (1 + jnp.sum(b2 / c2))
    Cminus = jnp.diag(1 / c2)
    Cminus_b = b / c2
    Sigma_inv = Cminus - alpha * (Cminus_b @ Cminus_b.T)

    A11_inv = (1 / (1 - alpha)) * ((1 - 1 / (jnp.sum(b2) + 1 - alpha)) * (b @ b.T) + jnp.diag(c2.flatten()))
    C = jnp.diag(c.flatten())

    A12 = 2 * (C @ Sigma_inv @ b @ jnp.ones((1, d))) * Sigma_inv
    A21 = A12.T
    A22 = 2 * C @ (Sigma_inv * Sigma_inv) @ C
    D = A22 - A21 @ A11_inv @ A12
    sol = jnp.linalg.lstsq(grad3, D)[0].T
    sol2 = jnp.linalg.lstsq(A21, D)[0]
    prod2 = A11_inv @ grad2 + (A11_inv @ A12) @ sol2 @ (A11_inv @ grad2) - (A11_inv @ A12) @ sol
    prod3 = -sol2 @ (A11_inv @ grad2) + sol
    return prod1, prod2, prod3


def grad_log_q_function(b, c, theta, mu):
    x = theta - mu
    d = b / c ** 2
    grad_log_q = -x / c ** 2 + (d.T @ x) / (1 + (d.T @ b)) * d
    return grad_log_q


def clip(X, threshold=100):
    # gradient clipping
    X_leaves = tree_leaves(X)
    norm = sum(tree_map(jnp.linalg.norm, X_leaves))

    def true_fun(x):
        return (threshold / norm) * x

    def false_fun(x):
        return x

    X = tree_map(lambda x: lax.cond(norm > threshold, true_fun, false_fun, x), X)
    return X


# To estimate the first term in lb = E_q(log f)-E_q(log q):lb_first_term

def vb_gauss_lowrank(key, logjoint_fn, data, nparams,
                     prior_mean=None, prior_std=0.01, init_scale=1., optimizer=optax.adam(1e-3),
                     num_samples=10, niters=5000, threshold=100, window_size=30, smooth=True):
    if prior_mean is None:
        mu_key, key = split(key, 2)
        mu = prior_std * normal(mu_key, shape=(nparams, 1))
    else:
        mu = prior_mean

    b_key, key = split(key, 2)
    b = prior_std * normal(b_key, shape=(nparams, 1))
    c = init_scale * jnp.ones((nparams, 1))

    variational_params = (mu, b, c)
    opt_state = optimizer.init(variational_params)

    def step_fn(variational_params, U_normal):
        mu, b, c = variational_params
        # Parameters in Normal distribution
        epsilon1 = U_normal[0]
        epsilon2 = U_normal[1:].reshape((-1, 1))
        theta = mu + b * epsilon1 + c * epsilon2
        h_theta, grad_h_theta = value_and_grad(logjoint_fn)(theta, data)
        # Gradient of  log variational distribution
        grad_log_q = grad_log_q_function(b, c, theta, mu)
        # Gradient of h(theta) and lower bound
        grad_theta = grad_h_theta - grad_log_q
        return variational_params, (grad_theta, epsilon1 * grad_theta, epsilon2 * grad_theta, h_theta)

    # Main VB loop
    def iter_fn(all_params, key):
        variational_params, opt_state = all_params
        mu, b, c = variational_params

        samples = normal(key, shape=(num_samples, nparams + 1))

        _, (*grad_lb_iter, lb_first_term) = lax.scan(step_fn, variational_params, samples)

        # Estimation of lower bound
        logdet = jnp.log(jnp.linalg.det(1 + (b / (c ** 2)).T @ b)) + jnp.sum(jnp.log((c ** 2)))

        # Mean of log-q -> mean(log q(theta))
        lb_log_q = -0.5 * nparams * jnp.log(2 * jnp.pi) - 0.5 * logdet - nparams / 2
        lb = jnp.mean(lb_first_term) - lb_log_q

        # Gradient of log variational distribution
        grad_lb = tree_map(lambda x: x.mean(axis=0), grad_lb_iter)
        grads = inverse_fisher_times_grad(b, c, grad_lb)

        # Gradient clipping
        grads = clip(grads, threshold=threshold)

        updates, opt_state = optimizer.update(grads, opt_state)
        variational_params = optax.apply_updates(variational_params, updates)

        return (variational_params, opt_state), (variational_params, lb)

    keys = split(key, niters)
    _, (variational_params, lower_bounds) = lax.scan(iter_fn, (variational_params, opt_state), keys)

    if smooth:
        def simple_moving_average(cur_sum, i):
            diff = (lower_bounds[i] - lower_bounds[i - window_size]) / window_size
            cur_sum += diff
            return cur_sum, cur_sum

        indices = jnp.arange(window_size, niters)
        cur_sum = jnp.sum(lower_bounds[:window_size]) / window_size
        _, lower_bounds = lax.scan(simple_moving_average, cur_sum, indices)
        lower_bounds = jnp.append(jnp.array([cur_sum]), lower_bounds)

    i = jnp.argmax(lower_bounds) + window_size - 1 if smooth else jnp.argmax(lower_bounds)
    best_params = tree_map(lambda x: x[i], variational_params)

    return best_params, lower_bounds

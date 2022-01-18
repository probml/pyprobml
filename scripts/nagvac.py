from jax.random import normal
import jax.numpy as jnp
import jax
from jax.random import split, PRNGKey


'''
TODO: Jaxify the implementation.
'''

def h_fn(w, batch, eps=0.01):
    X, y = batch
    p = 1 / (1 + jnp.exp(-jnp.dot(X, w)))
    p = jnp.clip(p, eps, 1 - eps)
    ll = jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
    lp = 0.5 * eps * jnp.sum(w**2)
    return ll + lp


def grad_log_q_function(b,c,theta,mu):
  x = theta - mu
  d = b /c**2
  grad_log_q = -x/c**2 + (d.T @ x) /(1+(d.T@b))*d
  return grad_log_q


## I^-1 x grad
def inverse_fisher_times_grad(b, c, grad):
    d = len(b)
    grad1, grad2, grad3 = grad
    c2 = c ** 2
    b2 = b ** 2
    prod1 = (b.T @ grad1) * b + (grad1 * c2)
    alpha = 1 / (1 + jnp.sum(b2 / c2))
    Cminus = jnp.diag(1 / c2)
    Cminus_b = b / c2
    Sigma_inv = Cminus - alpha * (Cminus_b @ Cminus_b.T)

    A11_inv = (1 / (1 - alpha)) * ((1 - 1 / (jnp.sum(b2) + 1 - alpha)) * (b * b.T) + jnp.diag(c2.flatten()))
    C = jnp.diag(c.flatten())

    A12 = 2 * (C @ Sigma_inv @ b @ jnp.ones((1, d))) @ Sigma_inv
    A21 = A12.T  # (7, 7)
    A22 = 2 * C @ (Sigma_inv * Sigma_inv) @ C  # (7, 7)
    D = A22 - A21 @ A11_inv @ A12  # (7, 7)
    sol = jnp.linalg.lstsq(grad3, D)[0].T
    sol2 = jnp.linalg.lstsq(A21, D)[0]
    prod2 = A11_inv @ grad2 + (A11_inv @ A12) @ sol2 @ (A11_inv @ grad2) - (A11_inv @ A12) @ sol
    prod3 = -sol2 @ (A11_inv @ grad2) + sol
    return jnp.concatenate([prod1, prod2, prod3])


# TODO: Add  labour force data to pyprobml data
# data = loadmat("/content/LabourForce.mat")['data']
X, y = jnp.array(data[:, :-1]), jnp.array(data[:, -1])

iter = 1
patience = 0
stop = False
LB_smooth = 0
lambda_best = []

# additional parameters that can be given the function as optional, initialize them none by default
ini_mu = None

# prior sigma for mu
std_init = 0.01

# Shape of mu, model params
d_theta = 7

# initial scale
init_scale = 0.1

# number of sample
S = 10

S = 200
max_patience = 20
max_iter = 100
max_grad = 200
window_size = 50
momentum_weight = 0.9

key = jax.random.PRNGKey(0)

'''
Initialization of mu
If initial parameters are not specified, then use some
initialization methods
'''

if ini_mu is None:
    mu_key, key = split(key, 2)
    mu = std_init * normal(mu_key, shape=(d_theta, 1))
else:
    mu = ini_mu

b_key, key = split(key, 2)
b = std_init * normal(b_key, shape=(d_theta, 1))
c = init_scale * jnp.ones((d_theta, 1))

# Variational parameters vector

lmbda = jnp.concatenate([mu, b, c])

tau_threshold = 2500
eps0 = 0.01  # learning_rate
# TODO: Store all setting to a structure
# param(iter,:) = mu.T

# First VB iteration
rqmc_key, key = split(key, 2)
rqmc = normal(rqmc_key, shape=(S, d_theta + 1))

# Store gradient of lb over S MC simulations
grad_lb_iter = jnp.zeros((S, 3 * d_theta))

# To estimate the first term in lb = E_q(log f)-E_q(log q)
lb_first_term = jnp.zeros((S, 1))


def init_fn(dummy, U_normal):
    # Parameters in Normal distribution
    epsilon1 = U_normal[0]
    epsilon2 = U_normal[1:].reshape((-1, 1))
    theta = mu + b * epsilon1 + c * epsilon2
    h_theta, grad_h_theta = jax.value_and_grad(h_fn)(theta, (X, y))
    # Gradient of  log variational distribution
    grad_log_q = grad_log_q_function(b, c, theta, mu)

    # Gradient of h(theta) and lowerbound
    grad_theta = grad_h_theta - grad_log_q;
    return None, (grad_theta, epsilon1 * grad_theta, epsilon2 * grad_theta, h_theta)


_, (*grad_lb_iter, lb_first_term) = jax.lax.scan(init_fn, None, rqmc)
# Estimation of lowerbound
logdet = jnp.log(jnp.linalg.det(1 + (b / (c ** 2)).T * b)) + jnp.sum(jnp.log((c ** 2)))
# Mean of log-q -> mean(log q(theta))
lb_log_q = -0.5 * d_theta * jnp.log(2 * jnp.pi) - 0.5 * logdet - d_theta / 2

LB = jnp.array([])
LB = jnp.append(LB, jnp.mean(lb_first_term) - lb_log_q)

# Gradient of log variational distribution
grad_lb = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_lb_iter)
gradient_lambda = inverse_fisher_times_grad(b, c, grad_lb)
gradient_bar = gradient_lambda

# Main VB loop
for i in range(max_iter):
    # if users want to save variational mean in each iteration
    # only use when debuging code
    '''TODO : Replace for loop with jax.lax.scan

    if(save_params):
        params_iter(iter,:) = mu'''

    iter = iter + 1;
    rqmc = normal(key, shape=(S, d_theta + 1))
    # store gradient of lb over S MC simulations
    grad_lb_iter = jnp.zeros((S, 3 * d_theta))
    # to estimate the first term in lb = E_q(log f)-E_q(log q)
    lb_first_term = jnp.zeros((S, 1))

    _, (*grad_lb_iter, lb_first_term) = jax.lax.scan(init_fn, None, rqmc)
    # Estimation of lowerbound
    logdet = jnp.log(jnp.linalg.det(1 + (b / (c ** 2)).T * b)) + jnp.sum(jnp.log((c ** 2)))
    # Mean of log-q -> mean(log q(theta))
    lb_log_q = -0.5 * d_theta * jnp.log(2 * jnp.pi) - 0.5 * logdet - d_theta / 2
    LB = jnp.append(LB, jnp.mean(lb_first_term) - lb_log_q)

    # Gradient of log variational distribution
    grad_lb = jax.tree_map(lambda x: x.mean(axis=0), grad_lb_iter)
    gradient_lambda = inverse_fisher_times_grad(b, c, grad_lb)

    # Gradient clipping
    grad_norm = jnp.linalg.norm(gradient_lambda)
    norm_gradient_threshold = max_grad

    if jnp.linalg.norm(gradient_lambda) > norm_gradient_threshold:
        gradient_lambda = (norm_gradient_threshold / grad_norm) * gradient_lambda

    gradient_bar = momentum_weight * gradient_bar + (1 - momentum_weight) * gradient_lambda

    if iter > tau_threshold:
        stepsize = eps0 * tau_threshold / iter
    else:
        stepsize = eps0

    # Reconstruct variantional parameters
    mu = mu + stepsize * gradient_bar[0] * stepsize
    b = b + stepsize * gradient_bar[1] * stepsize
    c = c + stepsize * gradient_bar[2] * stepsize

    '''
    TODO : Add smoothing as follows:

    if iter > window_size:  
        LB_smooth[iter-window_size] = jnp.mean(LB[iter-window_size+1:iter])
        if LB_smooth[end] >= jnp.max(LB_smooth):
            mu_best, b_best, c_best = mu, b, c
            patience = 0
        else:
            patience = patience + 1
    if (patience>max_patience) or (iter>max_iter):
        stop = True'''
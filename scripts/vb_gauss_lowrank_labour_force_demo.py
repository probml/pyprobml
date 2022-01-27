'''
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
max_iter = 500
max_grad = 200
window_size = 50
momentum_weight = 0.9

key = PRNGKey(0)


print(lmbda.shape)
tau_threshold = 2500
eps0 = 0.05  # learning_rate
# TODO: Store all setting to a structure
# param(iter,:) = mu.T
'''
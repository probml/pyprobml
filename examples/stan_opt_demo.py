#https://pystan.readthedocs.org/en/latest/optimizing.html
import pystan
import numpy as np
import matplotlib.pyplot as plt

ocode = """
data {
    int<lower=1> N;
    real y[N];
}
parameters {
    real mu;
}
model {
    y ~ normal(mu, 1);
}
"""

np.random.seed(1) 

sm = pystan.StanModel(model_code=ocode)
y2 = np.random.normal(size=20)
mle = np.mean(y2)

op = sm.optimizing(data=dict(y=y2, N=len(y2)))

'''
STAN OPTIMIZATION COMMAND (LBFGS)
init = random
save_iterations = 1
init_alpha = 0.001
tol_obj = 1e-12
tol_grad = 1e-08
tol_param = 1e-08
tol_rel_obj = 10000
tol_rel_grad = 1e+07
history_size = 5
seed = 884075560
initial log joint probability = -30.5022
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
       2      -11.9203       1.09052   1.11855e-14           1           1        4   
Optimization terminated normally: 
  Convergence detected: gradient norm is below tolerance
'''

  
op #OrderedDict([(u'mu', array(0.15616734214387562))])

assert(np.allclose(mle, op['mu']))
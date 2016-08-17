# 1d linear regression using SGD

import matplotlib.pyplot as plt
import numpy as np
import os
from linreg_1d_batch_demo import *
from utils import *
from optim_utils import *

init_lr = 0.01
n_steps = 200
batch_sizes = [10, N]
lr_decays = [1, 0.95]
momentums = [True]
folder = '/Users/kpmurphy/github/pmtk3/python/figures/'
n_expts = len(batch_sizes) * len(lr_decays) * len(momentums)

for batch_size in batch_sizes:
    for lr_decay in lr_decays:
        for momentum in momentums:
            np.random.seed(1)
            batchifier = Batchifier(Xtrain, ytrain, batch_size)
            params = np.zeros(2) 
            lr_fun = lambda(iter): get_learning_rate_exp_decay(iter, init_lr, lr_decay) 
            batch_size_frac = batch_size / np.float(N)
            ttl = 'batch={:0.2f}-lrdecay={:0.2f}-mom={}'.format(batch_size_frac, lr_decay, momentum)
            print 'starting experiment {}'.format(ttl)
            
            params, params_avg, obj_trace, params_trace, params_avg_trace = SGD(params,
                batchifier, n_steps,  get_objective, get_gradient, lr_fun,
                use_momentum=momentum, print_freq=20, store_params_trace=True)
          
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_loss_trace(obj_trace, loss_ols, ttl)
            fname = os.path.join(folder, 'linreg_1d_sgd_loss_trace_{}.png'.format(ttl))
            plt.savefig(fname)
            
            plot_error_surface_and_param_trace(xtrain, ytrain, w_true, params_trace, ttl)
            fname = os.path.join(folder, 'linreg_1d_sgd_param_trace_{}.png'.format(ttl))
            plt.savefig(fname)


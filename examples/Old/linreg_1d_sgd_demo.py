# 1d linear regression using SGD

#import numpy as np
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import os
from demos.linreg_1d_plot_demo import plot_error_surface_2d, plot_param_trace_2d, make_data_linreg_1d, plot_data_and_predictions_1d
import utils.util as util
import utils.sgd_util as sgd
from utils.optim_util import plot_loss_trace
from utils.linreg_model import LinregModel
from utils.mlp_model import MLP

def make_expt_config(N):
    batch_sizes = [N, 10]
    lr_decays = [0.99]
    momentums = [0, 0.9]
    init_lr = 0.005
    num_epochs = 20
    expt_config = []
    for batch_size in batch_sizes:
        for lr_decay in lr_decays:
            for momentum in momentums:
                config = {'batch_size': batch_size, 'lr_decay': lr_decay,
                    'momentum': momentum, 'optimizer': 'SGD',
                    'init_lr': init_lr, 'num_epochs': num_epochs, 'N': N}
                expt_config.append(config)
    return expt_config

def lr_fun_to_str(config):
    if not config.has_key('lr_fun'):
        return ''
    if config['lr_fun'] == 'exp':
         return 'LR-exp-init{:0.3f}-decay{:0.3f}'.format(config['init_lr'], config['lr_decay'])
    if config['lr_fun'] == 'const':
         return 'LR-const{:0.3f}'.format(config['init_lr'])

def config_to_str(config):
    cstr0 = '{}-{}'.format(config['fun_type'], config['model'])
    if config['optimizer'] == 'BFGS':
        cstr = cstr0 + '-BFGS'
    else:
        if config['batch_size'] == config['N']:
            batch_str = 'N'
        else:
            batch_str = '{}'.format(config['batch_size'])
        cstr_batch = 'batch{}'.format(batch_str)
        cstr_lr = lr_fun_to_str(config)
        if config['method'] == 'momentum':
            suffix = 'mom{}'.format(config['mass']) 
        if config['method'] == 'RMSprop':
            suffix = 'RMS{:0.3f}'.format(config['grad_sq_decay'])
        if config['method'] == 'ADAM':
            suffix = 'ADAM-{:0.3f}-{:0.3f}'.format(config['grad_decay'], config['grad_sq_decay'])
        if config['method'] == 'Rprop':
            suffix = 'Rprop'
        cstr = '-'.join([cstr0, cstr_batch, cstr_lr, suffix]) 
    return cstr
            
def main():
    np.random.seed(1)
    folder = 'figures/linreg-sgd'
    
    N = 50
    num_epochs = 100
   
    #fun_type = 'linear'
    fun_type = 'sine'
    #fun_type = 'quad'
    
    #model_type = 'linear'
    model_type = 'mlp:1-10-1'
                
    configs = []
    # BFGS has to be the first config, in order to compute loss_opt
    configs.append({'fun_type': fun_type, 'N': N, 'model': model_type, 
                    'optimizer': 'BFGS'})
                    
    configs.append({'fun_type': fun_type, 'N': N, 'model': model_type,  
                    'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs, 
                    'method': 'Rprop', 'improved': True})             
                                        
    configs.append({'fun_type': fun_type, 'N': N, 'model': model_type,  
                    'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs, 
                    'lr_fun': 'exp', 'init_lr': 0.05, 'lr_decay': 0.9,
                    'method': 'momentum', 'mass': 0.9})  
                    
    configs.append({'fun_type': fun_type, 'N': N, 'model': model_type,  
                    'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs, 
                    'lr_fun': 'exp', 'init_lr': 0.05, 'lr_decay': 0.9,
                    'method': 'RMSprop', 'grad_sq_decay': 0.9})
                    
    configs.append({'fun_type': fun_type, 'N': N, 'model': model_type,  
                    'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs, 
                    'lr_fun': 'exp', 'init_lr': 0.05, 'lr_decay': 0.9, 
                    'method': 'ADAM', 'grad_decay': 0.9, 'grad_sq_decay': 0.999})
    configs.append({'fun_type': fun_type, 'N': N, 'model': model_type,  
                    'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs, 
                    'lr_fun': 'const', 'init_lr': 0.05, 
                    'method': 'ADAM', 'grad_decay': 0.9, 'grad_sq_decay': 0.999})
    configs.append({'fun_type': fun_type, 'N': N, 'model': model_type,  
                    'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs, 
                    'lr_fun': 'const', 'init_lr': 0.001, 
                    'method': 'ADAM', 'grad_decay': 0.9, 'grad_sq_decay': 0.999})
    
    params_opt = None
    loss_opt = None
    for expt_num, config in enumerate(configs):
        np.random.seed(1)
        ttl = config_to_str(config)
        print '\nstarting experiment {}'.format(ttl)
        print config
        
        Xtrain, Ytrain, params_true, true_fun, fun_name = make_data_linreg_1d(\
            config['N'], config['fun_type'])
        data_dim = Xtrain.shape[1]
        
        if model_type == 'linear':
            model = LinregModel(data_dim, add_ones=True)
            params_opt, loss_opt = model.ols_fit(Xtrain, Ytrain)
        elif model_type[0:3] == 'mlp':
            _, layer_sizes = model_type.split(':')
            layer_sizes = [int(n) for n in layer_sizes.split('-')]
            model = MLP(layer_sizes, 'regression', L2_reg=0.001) 
        else:
             raise ValueError('unknown model type {}'.format(model_type))
                
        initial_params = model.init_params() 
        obj_fun = model.PNLL
        grad_fun = model.gradient
        
        param_dim = len(initial_params)
        plot_data = (data_dim == 1)
        plot_params = (param_dim == 2)
        nplots = 2
        if plot_data: 
            nplots += 1
        if plot_params:
            nplots += 1
        plot_rows, plot_cols = util.nsubplots(nplots)
         
        if config['optimizer'] == 'BFGS':
            logger = sgd.MinimizeLogger(obj_fun, grad_fun, (Xtrain, Ytrain), print_freq=1, store_params=True)
            params, loss, n_fun_evals = sgd.bfgs_fit(initial_params, obj_fun, grad_fun, (Xtrain, Ytrain), logger.update) 
            num_props = n_fun_evals * config['N']
            loss_avg = loss
            if params_opt is None:
                params_opt = params
                loss_opt = loss
                
        if config['optimizer'] == 'SGD':
            logger = sgd.SGDLogger(print_freq=20, store_params=True)
            if config.has_key('lr_fun'):
                if config['lr_fun'] == 'exp':
                    lr_fun = lambda iter, epoch: sgd.lr_exp_decay(iter, config['init_lr'], config['lr_decay']) 
                if config['lr_fun'] == 'const':
                    lr_fun = lambda iter, epoch: config['init_lr']
            else:
                lr_fun = None
                
            if config['method'] == 'momentum':
                sgd_updater = sgd.SGDMomentum(lr_fun, config['mass'])
            if config['method'] == 'RMSprop':
                sgd_updater = sgd.RMSprop(lr_fun, config['grad_sq_decay'])
            if config['method'] == 'ADAM':
                sgd_updater = sgd.ADAM(lr_fun, config['grad_decay'], config['grad_sq_decay'])
            if config['method'] == 'Rprop':
                sgd_updater = sgd.Rprop(improved_Rprop = config['improved'])
               
            params, loss, num_minibatch_updates, params_avg, loss_avg = sgd.sgd_minimize(initial_params, obj_fun, grad_fun,
                Xtrain, Ytrain, config['batch_size'], config['num_epochs'], sgd_updater, logger.update)
            num_props = num_minibatch_updates * config['batch_size']
           
                        
        
        print 'finished fitting, {} obj, {} grad, {} props'.format(model.num_obj_fun_calls, model.num_grad_fun_calls, num_props)

        fig = plt.figure()
        ax = fig.add_subplot(plot_rows, plot_cols, 1)
        plot_loss_trace(logger.obj_trace, loss_opt, ax, num_props)
        ax.set_title('final objective {:0.3f}, {:0.3f}'.format(loss, loss_avg))
        
        ax = fig.add_subplot(plot_rows, plot_cols, 2)
        ax.plot(logger.grad_norm_trace)
        ax.set_title('gradient norm vs num updates')
        
        if plot_data:
            ax = fig.add_subplot(plot_rows, plot_cols, 3)
            predict_fun = lambda X: model.predictions(params, X)
            plot_data_and_predictions_1d(Xtrain, Ytrain, true_fun, predict_fun, ax)
        
        if plot_params:
            ax = fig.add_subplot(plot_rows, plot_cols, 4)
            loss_fun = lambda w0, w1: model.PNLL([w0, w1], Xtrain, Ytrain)
            plot_error_surface_2d(loss_fun, params_opt, params_true, fun_type, ax)
            plot_param_trace_2d(logger.param_trace, ax)        
         
        fig.suptitle(ttl)        
        fname = os.path.join(folder, 'linreg_1d_sgd_{}.png'.format(ttl))
        plt.savefig(fname)
    
    plt.show()


if __name__ == "__main__":
    main()
 
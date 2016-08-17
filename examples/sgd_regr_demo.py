# Fit various 1d regression models using stochastic gradient descent

import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import demos.linreg_1d_plot_demo as demo
import utils.optim as opt
import utils.util as util
from utils.linreg_model import LinregModel
from utils.mlp_model import MLP

def lr_str(config):
    if config['lr_tune']==True:
        tune = '*'
    else:
        tune = ''
    if config['lr_step'] == np.inf:
        cstr = 'LR-{:0.4f}{}-const'.format(config['lr_init'], tune)
    else:
        #cstr = 'LR-exp-{:0.3f}{}-{:0.3f}-{:0.3f}'.format(config['lr_init'], tune, config['lr_decay'], config['lr_step'])
        cstr = 'LR-{:0.4f}{}-expdecay'.format(config['lr_init'], tune)
    return cstr

def batch_str(config):
    if config['batch_size'] == config['N']:
            batch_str = 'N'
    else:
        batch_str = '{}'.format(config['batch_size'])
    return 'batch{}'.format(batch_str)
    
def config_to_str(config):
    cstr0 = '{}-{}'.format(config['fun_type'], config['model_type'])
    if config['optimizer'] == 'BFGS':
        cstr = cstr0 + '-BFGS'
    else:
        cstr_batch = batch_str(config)
        cstr_lr = lr_str(config)
        cstr_sgd = config['name'] # config['sgd_fun'].__name__
        cstr = '-'.join([cstr0, cstr_sgd, cstr_lr, cstr_batch]) 
    return cstr

        
def run_expt(config, loss_opt=0):
    ttl = config_to_str(config)
    print '\nstarting experiment {}'.format(ttl)
    print config
    
    Xtrain, Ytrain, params_true, true_fun, fun_name = \
      demo.make_data_linreg_1d(config['N'], config['fun_type'])
    data_dim = Xtrain.shape[1]
    N = Xtrain.shape[0]
    Xtrain, Ytrain = opt.shuffle_data(Xtrain, Ytrain)
        
    model_type = config['model_type']
    if model_type == 'linear':
        model = LinregModel(data_dim, add_ones=True)
        params, loss = model.ols_fit(Xtrain, Ytrain)
    elif model_type[0:3] == 'mlp':
        _, layer_sizes = model_type.split(':')
        layer_sizes = [int(n) for n in layer_sizes.split('-')]
        model = MLP(layer_sizes, 'regression', L2_reg=0.001) 
    else:
        raise ValueError('unknown model type {}'.format(model_type))
            
    initial_params = model.init_params() 
    param_dim = len(initial_params)

    plot_data = (data_dim == 1)
    plot_params = (param_dim == 2)
    nplots = 1
    if plot_data: 
        nplots += 1
    if plot_params:
        nplots += 1
    plot_rows, plot_cols = util.nsubplots(nplots)



    if config['optimizer'] == 'BFGS':
        obj_fun = lambda params: model.PNLL(params, Xtrain, Ytrain)
        grad_fun = autograd.grad(obj_fun)
        logger = opt.OptimLogger(lambda params: obj_fun(params), eval_freq=1, 
                    store_freq=1, print_freq=1)   
        params, obj = opt.bfgs(obj_fun, grad_fun, initial_params, config['num_epochs'], 
                          logger.callback)
                         
    if config['optimizer'] == 'SGD':
        B = config['batch_size']
        M = N / B # num_minibatches_per_epoch (num iter per epoch)
        max_iters = config['num_epochs'] * M
        
        grad_fun_with_iter = opt.build_batched_grad(model.gradient, config['batch_size'], Xtrain, Ytrain)
        #obj_fun = opt.build_batched_grad(model.PNLL, config['batch_size'], Xtrain, Ytrain)
        obj_fun = lambda params: model.PNLL(params, Xtrain, Ytrain)
        sf = config.get('store_freq', M)
        logger = opt.OptimLogger(obj_fun, eval_freq=sf, store_freq=sf, print_freq=0)         
        sgd_fun = config['sgd_fun']
 
        if config['lr_tune']==True:
            eval_fun = lambda params: model.PNLL(params, Xtrain, Ytrain)
            lr, lrs, scores = opt.lr_tuner(eval_fun, 'grid', sgd_fun, grad_fun_with_iter,
                            initial_params, int(np.ceil(max_iters*0.1)))
            print 'lr tuner chose lr {:0.3f}'.format(lr)
            print lrs
            print scores
            config['lr_init'] = lr
            
        lr_fun = lambda iter: opt.lr_exp_decay(iter, config['lr_init'],
                                    config['lr_decay'], config['lr_step']) 
        params, obj = sgd_fun(obj_fun, grad_fun_with_iter, initial_params,
                            max_iters, logger.callback, lr_fun)
    
    training_loss = model.PNLL(params, Xtrain, Ytrain)
    print 'finished fitting, training loss {:0.3g}, {} obj calls, {} grad calls'.\
        format(training_loss, model.num_obj_fun_calls, model.num_grad_fun_calls)
    
    fig = plt.figure()
    ax = fig.add_subplot(plot_rows, plot_cols, 1)
    opt.plot_loss_trace(logger.eval_trace, loss_opt, ax)
    ax.set_title('final objective {:0.3g}'.format(training_loss))
    ax.set_xlabel('epochs')
    
    if plot_data:
        ax = fig.add_subplot(plot_rows, plot_cols, 2)
        predict_fun = lambda X: model.predictions(params, X)
        demo.plot_data_and_predictions_1d(Xtrain, Ytrain, true_fun, predict_fun, ax)
    
    if plot_params:
        ax = fig.add_subplot(plot_rows, plot_cols, 3)
        loss_fun = lambda w0, w1: model.PNLL(np.array([w0, w1]), Xtrain, Ytrain)
        demo.plot_error_surface_2d(loss_fun, params, params_true, config['fun_type'], ax)
        demo.plot_param_trace_2d(logger.param_trace, ax)        
        
    ttl = config_to_str(config) # recompute in case lr has been estimated
    fig.suptitle(ttl)
    folder = 'figures/linreg-sgd'        
    fname = os.path.join(folder, 'linreg_1d_sgd_{}.png'.format(ttl))
    plt.savefig(fname)
    return training_loss
  
        
              
def demo_linreg():
    N = 100
    num_epochs = 5
   
    #fun_type = 'linear-centered'
    fun_type = 'linear-uncentered'
    
    model_type = 'mlp:1-1'
                
    bfgs_config = {'fun_type': fun_type, 'N': N, 'model_type': model_type, 
                    'optimizer': 'BFGS', 'num_epochs': num_epochs}
    np.random.seed(1)           
    loss_opt = run_expt(bfgs_config)
           
    configs = []                                                                                    

    # LMS algorithm = SGD with batch size 1.
    def sgd_fun_vanilla_nobatch(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.sgd(obj_fun, grad_fun, x0, max_iters, callback,
                        lr_fun, mass=0, update='regular', avgdecay=0)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 1,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_vanilla_nobatch,  'lr_tune': False,
                'lr_init': 0.001, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'sgd-vanilla-nobatch', 'store_freq':1})

    # Need to set learning rate carefully!
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 1,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_vanilla_nobatch,  'lr_tune': False,
                'lr_init': 0.01, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'sgd-vanilla-nobatch', 'store_freq':1})

    # LR tuning
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 1,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_vanilla_nobatch,  'lr_tune': True,
                'lr_init': 0.01, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'sgd-vanilla-tuned', 'store_freq':1})
                
     # Minibatch helps
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_vanilla_nobatch,  'lr_tune': True,
                'lr_init': 0.01, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'sgd-vanilla-tuned-minibatch', 'store_freq':1})
       
    # Momentum helps         
    def sgd_fun_mom(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.sgd(obj_fun, grad_fun, x0, max_iters, callback,
                    lr_fun, mass=0.9, update='regular', avgdecay=0)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_mom,  'lr_tune': True,
                'lr_init': 0, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'sgd-mom', 'store_freq':1})
                
    # Nesterov helps
    def sgd_fun_nesterov(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.sgd(obj_fun, grad_fun, x0, max_iters, callback,
                    lr_fun, mass=0.9, update='nesterov', avgdecay=0)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_nesterov,  'lr_tune': True,
                'lr_init': 0, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'sgd-mom-nesterov', 'store_freq':1})
                         
    # Adam is often more robust  
    def sgd_fun_adam(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.adam(obj_fun, grad_fun, x0, max_iters, callback,
                    lr_fun,  avgdecay=0)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_adam,  'lr_tune': True,
                'lr_init': 0, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'adam', 'store_freq':1})
                
    # Averaging can help
    def sgd_fun_adam_avg(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.adam(obj_fun, grad_fun, x0, max_iters, callback,
                    lr_fun,  avgdecay=0.99)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_adam_avg,  'lr_tune': True,
                'lr_init': 0, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'adam-avg', 'store_freq':1})

                
    for expt_num, config in enumerate(configs):
        np.random.seed(1)
        run_expt(config, loss_opt)
      
    plt.show()



def demo_quad():
    N = 100
    num_epochs = 10
   
    fun_type = 'quad'
    
    model_type = 'mlp:1-10-10-1'
                
    bfgs_config = {'fun_type': fun_type, 'N': N, 'model_type': model_type, 
                    'optimizer': 'BFGS', 'num_epochs': num_epochs}
    np.random.seed(1)           
    loss_opt = run_expt(bfgs_config)
           
    configs = []                                                                                    


    # Momentum helps         
    def sgd_fun_mom(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.sgd(obj_fun, grad_fun, x0, max_iters, callback,
                    lr_fun, mass=0.9, update='regular', avgdecay=0)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_mom,  'lr_tune': True,
                'lr_init': 0, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'sgd-mom', 'store_freq':1})
                
    # Nesterov helps
    def sgd_fun_nesterov(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.sgd(obj_fun, grad_fun, x0, max_iters, callback,
                    lr_fun, mass=0.9, update='nesterov', avgdecay=0)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_nesterov,  'lr_tune': True,
                'lr_init': 0, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'sgd-mom-nesterov', 'store_freq':1})
                         
    # Adam is often more robust  
    def sgd_fun_adam(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.adam(obj_fun, grad_fun, x0, max_iters, callback,
                    lr_fun,  avgdecay=0)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_adam,  'lr_tune': True,
                'lr_init': 0, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'adam', 'store_freq':1})
                
    # Averaging can help
    def sgd_fun_adam_avg(obj_fun, grad_fun, x0, max_iters, callback, lr_fun):
                    return opt.adam(obj_fun, grad_fun, x0, max_iters, callback,
                    lr_fun,  avgdecay=0.99)
    configs.append({'fun_type': fun_type, 'N': N, 'model_type': model_type,  
                'optimizer': 'SGD', 'batch_size': 10,  'num_epochs': num_epochs,  
                'sgd_fun': sgd_fun_adam_avg,  'lr_tune': True,
                'lr_init': 0, 'lr_decay': 0.9, 'lr_step': 100,
                'name': 'adam-avg', 'store_freq':1})

                
    for expt_num, config in enumerate(configs):
        np.random.seed(1)
        run_expt(config, loss_opt)
      
    plt.show()




def main():
    demo_linreg()
    #demo_quad()

if __name__ == "__main__":
    main()
 

#https://github.com/IssamLaradji/stochastic_line_search/blob/master/sgd_armijo.py

import superimport

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class SGD_Armijo(torch.optim.Optimizer):
    def __init__(self,
                 model,
                 batch_size,
                 dataset_size,
                 init_step_size=1,
                 sigma=0.1,
                 beta=0.9,
                 beta_2=None):

        defaults = dict(
            batch_size=batch_size,
            init_step_size=init_step_size,
            dataset_size=dataset_size,
            sigma=sigma,
            beta=beta,
            beta_2=beta_2)

        super().__init__(model.parameters(), defaults)

        self.model = model
        self.state['step'] = 0
        self.state['step_size'] = init_step_size

    def step(self, closure):
        step_size = reset_step(self.state, self.defaults)

        # call the closure to get loss and compute gradients
        loss = closure()

        # save the current parameters:
        x_current = copy.deepcopy(self.param_groups)

        # save the gradient at the current parameters
        gradient, grad_norm = self.model.get_grads()

        # only do the check if the gradient norm is big enough
        with torch.no_grad():
            if grad_norm >= 1e-8:
                # check if condition is satisfied
                found = 0
                step_size_old = step_size

                for e in range(100):
                    # try a prospective step
                    self._try_update(step_size, x_current, gradient)

                    # compute the loss at the next step; no need to compute gradients.
                    loss_temp = closure(compute_grad=False)

                    wolfe_results = wolfe_line_search(step_size=step_size, 
                                                      step_size_old=step_size_old, 
                                                      loss=loss, 
                                                      grad_norm=grad_norm,
                                                      loss_temp=loss_temp, 
                                                      params=self.defaults)
                    
                    found, step_size, step_size_old = wolfe_results

                    if found == 1:
                        break

                if found == 0:
                    self._try_update(1e-6, x_current, gradient)

            else:
                self._try_update(step_size, x_current, gradient)

        # save the new step-size
        self.state['step_size'] = step_size
        self.state['step'] = self.state['step'] + 1

        return float(loss)

    def _try_update(self, step_size, x_current, gradient):
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                # update models parameters using SGD update
                p.data = x_current[i]['params'][j] - \
                    step_size * gradient[i][j]


# ==============================================
# Helpers


def wolfe_line_search(step_size, step_size_old, loss, grad_norm,
                      loss_temp, params):
    found = 0

    # computing the new break condition
    break_condition = loss_temp - \
        (loss - (step_size) * params['sigma'] * grad_norm**2)

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * params['beta']

    return found, step_size, step_size_old


def reset_step(state, params):
    step_size = state['step_size']

    if 'beta_2' in params and not params['beta_2'] is None:
        beta_2 = params['beta_2']
    else:
        beta_2 = 2.0

    # try to increase the step-size up to maximum ETA
    step_size = min(
        step_size * beta_2**(params['batch_size'] / params['dataset_size']),
        10.0)

    return step_size

###########

def get_grads(param_groups):
    grad_norm = 0
    gradient = []

    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    for i, group in enumerate(param_groups):
        grad_group = []
        for j, p in enumerate(group['params']):
            grad_copy = torch.zeros_like(p.grad.data).copy_(p.grad.data)
            grad_group.append(grad_copy)
            grad_norm = grad_norm + torch.sum(torch.mul(grad_copy, grad_copy))

        gradient.append(grad_group)

    return gradient, torch.sqrt(grad_norm)

class ArmijoModel(nn.Module):
    def __init__(self, model, objective):
        super().__init__()
        self.model = model
        self.objective = objective
        #self.opt = opt

    def forward(self, x):
        return self.model(x)

    def get_grads(self):
        return get_grads(list(self.parameters()))

    def step(self, batch):
        self.train()
        X, y = batch

        def closure(compute_grad=True):
            if compute_grad:
                self.zero_grad()
                
            logits = self.forward(X)
            loss = self.objective(logits, y)

            if compute_grad:
                loss.backward()

            return float(loss)

        minibatch_loss = self.opt.step(closure)
        
        return float(minibatch_loss) , self.opt.state['step_size']



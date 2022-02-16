
''' 
This script is based on https://github.com/chenzeno/FOO-VB/blob/ebc14a930ba9d1c1dadc8e835f746c567c253946/main.py

'''
import numpy as np

from time import time

from jax import random, value_and_grad, tree_map, vmap, lax
import jax.numpy as jnp

from functools import partial

import foo_vb_utils_copy as utils

def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)

def init_step(key, model, image_size, config):
    model_key, param_key = random.split(key) 
    variables = model.init(model_key, jnp.zeros((config.batch_size, image_size)))
    params = tree_map(jnp.transpose, variables)
    lists = utils.init_param(param_key, params, config.s_init, True, config.alpha)
    return lists


def train_step(key, lsts, data, target, value_and_grad_fn, train_mc_iters, eta, diagonal):
        w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = lsts
        
        def monte_carlo_step(agg_lsts, key):
            # Phi ~ MN(0,I,I)
            avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = agg_lsts
            phi_key, key = random.split(key)
            phi_mat_lst = utils.gen_phi(phi_key, w_mat_lst)
            
            # W = M +B*Phi*A^t
            params = utils.randomize_weights(m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
            loss, grads = value_and_grad_fn(tree_map(jnp.transpose, params), data, target)
            grad_mat_lst = utils.weight_grad(grads)
            avg_psi_mat_lst = utils.aggregate_grads(avg_psi_mat_lst, grad_mat_lst, train_mc_iters)
            e_a_mat_lst = utils.aggregate_e_a(e_a_mat_lst, grad_mat_lst, b_mat_lst,
                            phi_mat_lst, train_mc_iters)
            
            e_b_mat_lst = utils.aggregate_e_b(e_b_mat_lst, grad_mat_lst, a_mat_lst,
                                phi_mat_lst, train_mc_iters)
            
            return (avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst), loss

        # M = M - B*B^t*avg_Phi*A*A^t
        keys = random.split(key, train_mc_iters)
        (avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst), losses = scan(monte_carlo_step, (avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst), keys) 
        
        print("Loss :", losses.mean())
        
        m_mat_lst = utils.update_m(m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, eta, diagonal=diagonal)
        a_mat_lst, b_mat_lst = utils.update_a_b(a_mat_lst, b_mat_lst, e_a_mat_lst, e_b_mat_lst)
        avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = utils.zero_matrix(avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst)
        
        lists = w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst
        
        return lists, losses


def eval_step(model, lsts, data, target, train_mc_iters):
    w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = lsts 

    def monte_carlo_step(w_mat_lst, phi_key):
        phi_mat_lst = utils.gen_phi(phi_key, w_mat_lst)
        params = utils.randomize_weights(m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
        output = model.apply(tree_map(jnp.transpose, params), data)
        # get the index of the max log-probability
        pred = jnp.argmax(output, axis=1)
        return w_mat_lst, jnp.sum(pred==target)

    keys = random.split(random.PRNGKey(0), train_mc_iters)
    _, correct_per_iter = scan(monte_carlo_step,  w_mat_lst, keys)
    n_correct = jnp.sum(correct_per_iter)

    return n_correct
    

def train_continuous_mnist(key, model, train_loader,
                           test_loader, image_size, num_classes, config):

    init_key, key = random.split(key)
    lists = init_step(key, model, image_size, config)
    criterion = partial(utils.cross_entropy_loss,
                        num_classes=num_classes,
                        predict_fn=model.apply)
    
    grad_fn = value_and_grad(criterion)

    ava_test = [] 

    for task in range(len(test_loader)):
        for epoch in range(1, config.epochs + 1):
            start_time = time()
            for batch_idx, (data, target) in enumerate(train_loader[0]):
                data, target = jnp.array(data.view(-1, image_size).numpy()), jnp.array(target.numpy())

                train_key, key = random.split(key) 
                lists, losses = train_step(train_key, lists, data, target, grad_fn,
                                     config.train_mc_iters, config.eta, config.diagonal)
            
            print("Time : ", time() - start_time)
            total = 0

            for data, target in test_loader[task]:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                n_correct = eval_step(model, lists, data, target, config.train_mc_iters)
                total += n_correct
                
            test_acc = 100. * total / (len(test_loader[task].dataset) * config.train_mc_iters)
            print('\nTask num {}, Epoch num {} Test Accuracy: {:.2f}%\n'.format(
                task, epoch, test_acc))
        
        test_acc_lst = []
        
        for i in range(task + 1):
            total = 0
            for data, target in test_loader[i]:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                n_correct = eval_step(model, lists, data, target, config.train_mc_iters)
                total += n_correct
            
            test_acc = 100. * total / (len(test_loader[task].dataset) * config.train_mc_iters)
            test_acc_lst.append(test_acc)
            
            print('\nTraning task Num: {} Test Accuracy of task {}: {:.2f}%\n'.format(
                task, i, test_acc))
        ava_test.append(jnp.mean(test_acc_lst))
    
    return ava_test


def train_multiple_tasks( key, model, train_loader,
                           test_loader, num_classes,
                           perm_lst, image_size, config):
    
    init_key, key = random.split(key)
    lists = init_step(key, model, config)    
    criterion = partial(utils.cross_entropy_loss,
                        num_classes=num_classes, predict_fn=model.apply)
    
    grad_fn = value_and_grad(criterion)
    
    ava_test = []

    for task in range(len(perm_lst)):
        for epoch in range(1, config.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):                
                data, target = jnp.array(data.detach().numpy().reshape((-1, image_size))), jnp.array(target.detach().numpy())
                data = data[:, perm_lst[task]]

                train_key, key = random.split(key)
                start_time = time.time()
                lists, losses = train_step(train_key, lists, data, target, grad_fn,
                                    config.train_mc_iters, config.eta, config.diagonal)
                print("Time : ", start_time - time.time())
            
            total = 0
            
            for data, target in train_loader:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                data = data[:, perm_lst[task]]
                n_correct = eval_step(model, lists, data, target, config.train_mc_iters)
                total += n_correct

            train_acc = 100. * total / (len(train_loader.dataset) * args.train_mc_iters)

            total = 0

            for data, target in test_loader:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                data = data[:, perm_lst[task]]
                n_correct = eval_step(model, lists, data, target, config.train_mc_iters)
                total += n_correct

            test_acc = 100. * total / (len(test_loader.dataset) * config.train_mc_iters) 
            print('\nTask num {}, Epoch num {}, Train Accuracy: {:.2f}% Test Accuracy: {:.2f}%\n'.format(
                task, epoch, train_acc, test_acc))

        test_acc_lst = []
        
        for i in range(task + 1):
            total = 0

            for data, target in test_loader:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                data = data[:, perm_lst[i]]

                n_correct = eval_step(model, lists, data, target, config.train_mc_iters)
                total += n_correct
            
            test_acc = 100. * total / (len(test_loader.dataset) * args.train_mc_iters)
            test_acc_lst.append(test_acc)
            print('\nTraning task Num: {} Test Accuracy of task {}: {:.2f}%\n'.format(
                task, i, test_acc))
        
        print(test_acc_lst)
        ava_test.append(jnp.mean(test_acc_lst))
        return ava_test

''' 
Foo
This script is based on https://github.com/chenzeno/FOO-VB/blob/ebc14a930ba9d1c1dadc8e835f746c567c253946/main.py

'''
import numpy as np

from jax import random, value_and_grad, tree_map, vmap, lax
import jax.numpy as jnp

from functools import partial

import foo_vb_utils as utils

def train_continuous_mnist(key, model, train_loader,
                           test_loader, num_classes, config):
    ava_test = [] 

    init_key, key = random.split(key)
    variables = model.init(init_key, jnp.zeros((config.batch_size, 784)))
    params = tree_map(jnp.transpose, variables)
    init_key, key = random.split(key)
    lists = utils.init_param(init_key, params, config.s_init, True, config.alpha)
    w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = lists
    
    criterion = partial(utils.cross_entropy_loss,
                        num_classes=num_classes, predict_fn=model.apply)
    
    grad_fn = value_and_grad(criterion)

    for task in range(len(test_loader)):
        for epoch in range(1, config.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader[0]):
                data, target = jnp.array(data.detach().numpy()), jnp.array(target.detach().numpy())
                data = data.reshape((-1, 784))
                
                for mc_iter in range(config.train_mc_iters):
                    # Phi ~ MN(0,I,I)
                    phi_key, key = random.split(key)
                    
                    phi_mat_lst = utils.gen_phi(phi_key, w_mat_lst)
                    
                    # W = M +B*Phi*A^t
                    params, w_mat_lst = utils.randomize_weights(params, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                    loss, grads = grad_fn(tree_map(jnp.transpose, params), data, target)
                    grad_mat_lst = utils.weight_grad(grads)
                    avg_psi_mat_lst = utils.aggregate_grads(avg_psi_mat_lst, grad_mat_lst, config.train_mc_iters)
                    e_a_mat_lst = utils.aggregate_e_a(e_a_mat_lst, grad_mat_lst, b_mat_lst,
                                  phi_mat_lst, config.train_mc_iters)
                    
                    e_b_mat_lst = utils.aggregate_e_b(e_b_mat_lst, grad_mat_lst, a_mat_lst,
                                        phi_mat_lst, config.train_mc_iters)

                # M = M - B*B^t*avg_Phi*A*A^t
                m_mat_lst = utils.update_m(m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, config.eta)
                a_mat_lst, b_mat_lst = utils.update_a_b(a_mat_lst, b_mat_lst, e_a_mat_lst, e_b_mat_lst)
                avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = utils.zero_matrix(avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst)
            correct = 0
            for data, target in test_loader[task]:
                data, target = jnp.array(data.numpy()), jnp.array(target.numpy())
                data = data.reshape((-1, 784))
                
                def monte_carlo_iter_fn(all_params, phi_key):
                    params, w_mat_lst = all_params
                    phi_mat_lst = utils.gen_phi(phi_key, w_mat_lst)
                    params, w_mat_lst = utils.randomize_weights(params, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                    output = model.apply(tree_map(jnp.transpose, params), data)
                    pred = jnp.argmax(output, axis=1) # keepdims=True)  # get the index of the max log-probability
                    return (params, w_mat_lst), jnp.sum(pred==target)

                def scan(f, init, xs, length=None):
                    if xs is None:
                        xs = [None] * length
                    carry = init
                    ys = []
                    for x in xs:
                        carry, y = f(carry, x)
                        ys.append(y)
                    return carry, jnp.stack(ys)

                keys = random.split(random.PRNGKey(0), config.train_mc_iters)
                (params, w_mat_lst), correct_per_task = scan(monte_carlo_iter_fn, (params, w_mat_lst), keys)
                correct += jnp.sum(correct_per_task)

            test_acc = 100. * correct / (len(test_loader[task].dataset) * config.train_mc_iters)
            print('\nTask num {}, Epoch num {} Test Accuracy: {:.2f}%\n'.format(
                task, epoch, test_acc))
        
        test_acc_lst = []
        for i in range(task + 1):
            correct = 0
            for data, target in test_loader[i]:
                data, target = jnp.array(data.numpy()), jnp.array(target.numpy())
                data = data.reshape((-1, 784))
                for mc_iter in range(config.train_mc_iters):
                    phi_key, key = random.split(key)
                    phi_mat_lst = utils.gen_phi(w_mat_lst)
                    params, w_mat_lst = utils.randomize_weights(params, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                    output = model(data)
                    pred = jnp.argmax(axis=1) # get the index of the max log-probability
                    correct += jnp.sum(pred == target)
            test_acc = 100. * correct / (len(test_loader[i].dataset) * config.train_mc_iters)
            test_acc_lst.append(test_acc)
            print('\nTraning task Num: {} Test Accuracy of task {}: {:.2f}%\n'.format(
                task, i, test_acc))
        print(test_acc_lst)
        ava_test.append(np.average(np.asanyarray(test_acc_lst)))
    return ava_test


def train_multiple_tasks( key, model, train_loader,
                           test_loader, num_classes,
                           perm_lst, config):

    ava_test = []
    
    init_key, key = random.split(key)
    variables = model.init(init_key, jnp.zeros((config.batch_size, num_classes)))
    params = variables["params"]

    criterion = partial(utils.cross_entropy_loss,
                            num_classes=num_classes, predict_fn=model.apply)

    lists = utils.init_param(params, config.s_init, True, config.alpha)
    w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = lists
    
    criterion = partial(utils.cross_entropy_loss,
                        num_classes=num_classes, predict_fn=model.apply)
    
    grad_fn = value_and_grad(criterion)

    
    for task in range(len(perm_lst)):
        for epoch in range(1, config.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                model.train()
                data, target = jnp.array(data.numpy()), jnp.array(target.numpy())
                data = data.reshape(-1, 784)
                data = data[:, perm_lst[task]]
                for mc_iter in range(config.train_mc_iters):
                    # Phi ~ MN(0,I,I)
                    phi_mat_lst = utils.gen_phi(w_mat_lst)
                    # W = M +B*Phi*A^t
                    params, w_mat_lst = utils.randomize_weights(params, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                    output = model(data)
                    print(loss)
                    loss.backward()
                    grad_mat_lst = grad_fn(params, data, target)
                    
                    avg_psi_mat_lst = utils.aggregate_grads(avg_psi_mat_lst, grad_mat_lst, config.train_mc_iters)
                    e_a_mat_lst = utils.aggregate_e_a(e_a_mat_lst, grad_mat_lst, b_mat_lst,
                    phi_mat_lst, config.train_mc_iters)
                    
                    e_b_mat_lst = utils.aggregate_e_b(e_b_mat_lst, grad_mat_lst, a_mat_lst,
                                        phi_mat_lst, config.train_mc_iters)
                # M = M - B*B^t*avg_Phi*A*A^t
                m_mat_lst = utils.update_m(m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, config.eta)  # , task == 0)
                a_mat_lst, b_mat_lst = utils.update_a_b(a_mat_lst, b_mat_lst, e_a_mat_lst, e_b_mat_lst)
                utils.zero_matrix(avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst)

                correct = 0
                for data, target in train_loader:
                    data, target = jnp.array(data.numpy()), jnp.array(target.numpy())
                    data = data.reshape(-1, 784)
                    data = data[:, perm_lst[task]]
                    for mc_iter in range(config.train_mc_iters):
                        phi_mat_lst = utils.gen_phi(w_mat_lst)
                        params, w_mat_lst = utils.randomize_weights(params, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                train_acc = 100. * correct / (len(train_loader.dataset) * config.train_mc_iters)
                correct = 0
                for data, target in test_loader:
                    data, target = jnp.array(data.numpy()), jnp.array(target.numpy())
                    data = data.view(-1, 784)
                    data = data[:, perm_lst[task]]
                    for mc_iter in range(config.train_mc_iters):
                        phi_mat_lst = utils.gen_phi(w_mat_lst)
                        params, w_mat_lst = utils.randomize_weights(params, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    break
                test_acc = 100. * correct / (len(test_loader.dataset) * config.train_mc_iters)
                print(test_acc)
            print('\nTask num {}, Epoch num {}, Train Accuracy: {:.2f}% Test Accuracy: {:.2f}%\n'.format(
                task, epoch, train_acc, test_acc))
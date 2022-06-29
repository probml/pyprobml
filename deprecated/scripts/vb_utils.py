'''
Consists of functions that can be used for any variational bayes algorithm.
Author: Aleyna Kara(@karalleyna)
'''

from jax import tree_leaves, tree_map, lax
import jax.numpy as jnp


def clip(X, threshold):
    '''
    Clips the
    Parameters
    ----------
    X : pytree

    threshold : float
        If the norm is above the threshold, then X  will be updated as follows:
            X = (threshold / norm ) * X
    Returns
    -------
    pytree : Updated X
    '''
    X_leaves = tree_leaves(X)
    norm = sum(tree_map(jnp.linalg.norm, X_leaves))

    def true_fun(x):
        return (threshold / norm) * x

    def false_fun(x):
        return x

    X = tree_map(lambda x: lax.cond(norm > threshold, true_fun, false_fun, x), X)
    return X

# Implement inference in a Bayes net using einsum and AD
# Based on "A differential approach to inference in Bayesian networks"
# Adnan Darwiche, JACM 2003

import numpy as onp # original numpy
import jax.numpy as np
from jax import grad
from jax.ops import index, index_add, index_update

def make_evidence_vector(nstates, val):
    '''Create delta function on observed value. Use val=-1 if hidden.'''
    if val == -1:
        lam = np.ones(nstates) 
    else: 
        #lam[val] = 1.0 # not allowed to mutate state in jax
        lam = index_update(np.zeros(nstates), index[val], 1.0) # functional assignment
    return lam
    
def make_evidence_vectors(cardinality, evidence):
    return {name: make_evidence_vector(nstates, evidence.get(name, -1)) for name, nstates in cardinality.items()}

def make_einsum_string(parents):
    # example: 'A,B,C,  A,BA,CA->' for B <- A -> C
    node_names = list(parents.keys())
    cpt_names = [n + ''.join(parents[n]) for n in node_names] # indices for CPTs
    str = ','.join(node_names) + ',' + ','.join(cpt_names) + '->'
    return str
    
def network_poly(dag, params, evectors):
    str = make_einsum_string(dag)
    # Extract dictionary elements in same order as string
    node_names = list(dag.keys())
    evecs = []
    cpts = []
    for n in node_names:
        evecs.append(evectors[n])
        cpts.append(params[n])
    # Sum over all assignments to network polynomial
    return np.einsum(str, *(evecs+cpts))


def marginal_probs(dag, params, evidence):
    '''Compute marginal probabilities of all nodes in a Bayesnet.
    '''
    #cardinality = [np.shape(CPT)[0] for CPT in params]
    cardinality = {name: np.shape(CPT)[0] for name, CPT in params.items()}
    evectors = make_evidence_vectors(cardinality, evidence)
    f = lambda ev: network_poly(dag, params, ev) # clamp model parameters
    prob_ev = f(evectors) 
    grads = grad(f)(evectors) # list of derivatives wrt evectors
    probs = dict()
    for name in dag.keys():
        ev = evidence.get(name, -1)
        if ev == -1:
            probs[name] = grads[name] / prob_ev  # corollary 1 of Darwiche03
        else:
            probs[name] = evectors[name] # clamped node
    return prob_ev, probs

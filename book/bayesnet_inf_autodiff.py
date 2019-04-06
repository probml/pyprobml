# Implements inference in a Bayes net using autodiff applied to Z=einsum(factors).
# murphyk@gmail.com, April 2019

# Based on "A differential approach to inference in Bayesian networks"
# Adnan Darwiche, JACM 2003.
# http://
#
# Darwiche defines the network polynomial f(l) = poly(l, theta),
# where l(i,j)=1 if variable i is in state j; these
# are called  the evidence vectors, denoted by lambda. 
# Let e be a vector of observations for a (sub)set of the nodes.
# Let o(i)=1 if variable i is observed in e, and o(i)=0 otherwise.
# So l(i,j)=ind{j=e(i)) if o(i)=1, and l(i,:)=1 otherwise.
# Thus l(i,j)=1 means the setting x(i)=j is compatible with e.
# Define f(e) = f(l(e)), where l(e) is this binding process.
#
# Thm 1: f(l(e)) = Pr(x(o)=e) = Pr(e), the probability of the evidence.
# f(e) is also denoted by Z, the normalization constant in Bayes rule.
# Note that we can compute f(e) using einstein summation over all terms
# in the network poly.
#
# Thm 2: let g_{i,j}(l)= d/dl(i,j) f(l(e)) be partial derivative.
# so g_i(l) is the gradient vector for variable i.
# Then g_ij(l) = Pr(x(i)=j, x(o(-i))=e(-i)), where o(-i) are all observed
# variables except i.
#
# Corollary 1. d/dl(i,j) log f(l(e)) = 1/f(l(e)) * g_{ij}(l(e))
#   = Pr(x(i)=j | e) if o(i)=0 (so i not in e). 
# This is the standard result that derivatives of the log partition function
# gives the expected sufficient statistics, which for a multinimomial
# are the posterior marignals over states.
#
# A similar result is shown in 
# "Inside-outside and forward-backward algorithms are just backprop",
# Jason Eisner (2016).
# EMNLP Workshop on Structured Prediction for NLP. 
# http://cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf
#
# We use jax (https://github.com/google/jax) to compute the partial
# derivatives. This requires that f(e) be implemented using jax's
# version of einsum, which fortunately is 100% compatible with the numpy
# version.


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
    def f(nstates, val):
         if val == -1:
             lam = np.ones(nstates) 
         else: 
             #lam[val] = 1.0 # not allowed to mutate state in jax
             lam = index_update(np.zeros(nstates), index[val], 1.0) # functional assignment
         return lam
    return {name: f(nstates, evidence.get(name, -1)) for name, nstates in cardinality.items()}

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

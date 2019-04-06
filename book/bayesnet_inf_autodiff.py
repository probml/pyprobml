'''
Implements inference in a Bayes net using autodiff applied to Z=einsum(factors).
 murphyk@gmail.com, April 2019

 Based on "A differential approach to inference in Bayesian networks"
 Adnan Darwiche, JACM 2003.
 Cached copy:
 https://github.com/probml/pyprobml/blob/master/data/darwiche-acm2003-differential.pdf


 A similar result is shown in 
 "Inside-outside and forward-backward algorithms are just backprop",
 Jason Eisner (2016).
 EMNLP Workshop on Structured Prediction for NLP. 
 http://cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf
 
 For a demo of how to use this code, see
 https://github.com/probml/pyprobml/blob/master/book/student_pgm_inf_autodiff.py
 For a unit test see
 https://github.com/probml/pyprobml/blob/master/book/bayesnet_inf_autodiff_test.py
 
 Darwiche defines the network polynomial f(l) = poly(l, theta),
 where l(i,j)=1 if variable i is in state j; these
 are called  the evidence vectors, denoted by lambda. 
 Let e be a vector of observations for a (sub)set of the nodes.
 Let o(i)=1 if variable i is observed in e, and o(i)=0 otherwise.
 So l(i,j)=ind{j=e(i)) if o(i)=1, and l(i,:)=1 otherwise.
 Thus l(i,j)=1 means the setting x(i)=j is compatible with e.
 Define f(e) = f(l(e)), where l(e) is this binding process.

 Thm 1: f(l(e)) = Pr(x(o)=e) = Pr(e), the probability of the evidence.
 f(e) is also denoted by Z, the normalization constant in Bayes rule.
 Note that we can compute f(e) using einstein summation over all terms
 in the network poly.

 Thm 2: let g_{i,j}(l)= d/dl(i,j) f(l(e)) be partial derivative.
 so g_i(l) is the gradient vector for variable i.
 Then g_ij(l) = Pr(x(i)=j, x(o(-i))=e(-i)), where o(-i) are all observed
 variables except i.

 Corollary 1. d/dl(i,j) log f(l(e)) = 1/f(l(e)) * g_{ij}(l(e))
   = Pr(x(i)=j | e) if o(i)=0 (so i not in e). 
 This is the standard result that derivatives of the log partition function
 gives the expected sufficient statistics, which for a multinomial
 are the posterior marignals over states.


 We use jax (https://github.com/google/jax) to compute the partial
 derivatives. This requires that f(e) be implemented using jax's
 version of einsum, which fortunately is 100% compatible with the numpy
 version.
 '''


import numpy as onp # original numpy
import jax.numpy as np
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update
from functools import partial
    
def make_einsum_string(dag):
    # example: if dag is  B <- A -> C, returns 'A,B,C,  A,BA,CA->' 
    node_names = list(dag.keys())
    cpt_names = [n + ''.join(dag[n]) for n in node_names] # indices for CPTs
    str = ','.join(node_names) + ',' + ','.join(cpt_names) + '->'
    return str

def make_list_of_factors(dag, params, evectors):
    # Extract dictionary elements in same order as einsum string
    node_names = list(dag.keys())
    evecs = []
    cpts = []
    for n in node_names:
        evecs.append(evectors[n])
        cpts.append(params[n])
    return (evecs+cpts)
    
def network_poly(dag, params, evectors, elim_order=None):
    # Sum over all assignments to network polynomial to compute Z
    str = make_einsum_string(dag)
    factors = make_list_of_factors(dag, params, evectors)
    if elim_order is None:
        return np.einsum(str, *factors)
    else:
        return np.einsum(str, *factors, optimize=elim_order)

def make_evidence_vectors(cardinality, evidence):
    # compute l(i,j)=1 iff x(i)=j is compatible with evidence e
    def f(nstates, val):
         if val == -1:
             vec = np.ones(nstates) 
         else: 
             #vec[val] = 1.0 # not allowed to mutate state in jax
             vec = index_update(np.zeros(nstates), index[val], 1.0) # functional assignment
         return vec
    return {name: f(nstates, evidence.get(name, -1)) for name, nstates in cardinality.items()}

def marginal_probs(dag, params, evidence, elim_order=None):
    # Compute marginal probabilities of all nodes in a Bayesnet.
    cardinality = {name: np.shape(CPT)[0] for name, CPT in params.items()}
    evectors = make_evidence_vectors(cardinality, evidence)
    f = lambda ev: network_poly(dag, params, ev, elim_order) # clamp model parameters
    prob_ev = f(evectors) 
    grads = grad(f)(evectors) # list of derivatives wrt evectors
    probs = dict()
    for name in dag.keys():
        ev = evidence.get(name, -1)
        if ev == -1: # not observed
            probs[name] = grads[name] / prob_ev  
        else:
            probs[name] = evectors[name] # clamped node
        probs[name] = onp.array(probs[name]) # cast back to vanilla numpy array
    return prob_ev, probs

def marginals_batch(dag, params, evidence_list, elim_order=None):
    f = lambda ev:  marginal_probs(dag, params, ev, elim_order)
    return vmap(f)(evidence_list)
    
    
def compute_elim_order(dag, params):
    # compute optimal elimination order assuming no nodes are observed
    evidence = {}
    cardinality = {name: np.shape(CPT)[0] for name, CPT in params.items()}
    evectors = make_evidence_vectors(cardinality, evidence)
    str = make_einsum_string(dag)
    factors = make_list_of_factors(dag, params, evectors)
    elim_order = np.einsum_path(str, *factors, optimize='optimal')[0]
    return elim_order
        
# Class that provides syntactic sugar on top of above functions.
class BayesNetInfAutoDiff:
    def __init__(self, dag, params):
        self._dag = dag
        self._params = params
        # unfortunately jax.np.einsum_path is not yet implemented
        #self._elim_order = compute_elim_order(dag, params)
        self._elim_order = None
        
    def infer_marginals(self, evidence):
        prob_ev, marginals = marginal_probs(self._dag, self._params, evidence,
                                        self._elim_order)
        return marginals
    
    def infer_prob_evidence(self, evidence):
        prob_ev, marginals = marginal_probs(self._dag, self._params, evidence,
                                        self._elim_order)
        return prob_ev
    
    

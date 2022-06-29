
import superimport

import numpy as np # original numpy
import jax.numpy as jnp
from jax import grad
import bayesnet_inf_autodiff as bn

# Example from fig 3 of Darwiche'03 paper
# Note that we assume 0=False, 1=True so we order the entries differently

thetaA = jnp.array([0.5, 0.5]) # thetaA[a] = P(A=a)
thetaB = jnp.array([[1.0, 0.0], [0.0, 1.0]]) # thetaB[b,a] = P(B=b|A=a)
thetaC = jnp.array([[0.8, 0.2], [0.2, 0.8]]) # thetaC[c,a] = P(C=c|A=a)
params = {'A': thetaA, 'B': thetaB, 'C':thetaC}

cardinality = {name: jnp.shape(cpt)[0] for name, cpt in params.items()}

dag = {'A':[], 'B':['A'], 'C':['A']}

assert bn.make_einsum_string(dag) == 'A,B,C,A,BA,CA->'
          
#evidence = [1, None, 0] # a=T, c=F
evidence = {'A':1, 'C':0}

evectors = bn.make_evidence_vectors(cardinality, evidence)
fe = bn.network_poly(dag, params, evectors) # probability of evidence
assert fe==0.1

# compare numbers to table 1 of Darwiche03
f = lambda ev: bn.network_poly(dag, params, ev)
grads = grad(f)(evectors) # list of derivatives wrt evectors
assert jnp.allclose(grads['A'], [0.4, 0.1]) # A
assert jnp.allclose(grads['B'], [0.0, 0.1]) # B 
assert jnp.allclose(grads['C'], [0.1, 0.4]) # C

prob_ev, probs = bn.marginal_probs(dag, params, evidence)
assert prob_ev==0.1
assert jnp.allclose(probs['B'], [0.0, 1.0])

print('tests passed')
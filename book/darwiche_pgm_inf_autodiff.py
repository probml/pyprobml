# Inference in a Bayes net using einsum and AD
# Based on "A differential approach to inference in Bayesian networks"
# Adnan Darwiche, JACM 2003

import numpy as onp # original numpy
import jax.numpy as np
from jax import grad

import bayesnet_inf_autodiff as bn

# Example from fig 3 of Darwiche'03 paper
# Note that we assume 0=False, 1=True so we order the entries differently

thetaA = np.array([0.5, 0.5]) # thetaA[a] = P(A=a)
thetaB = np.array([[1.0, 0.0], [0.0, 1.0]]) # thetaB[b,a] = P(B=b|A=a)
thetaC = np.array([[0.8, 0.2], [0.2, 0.8]]) # thetaC[c,a] = P(C=c|A=a)
#params = [thetaA, thetaB, thetaC]
params = {'A': thetaA, 'B': thetaB, 'C':thetaC}

#cardinality = [2,2,2]
cardinality = {name: np.shape(cpt)[0] for name, cpt in params.items()}

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
assert np.allclose(grads['A'], [0.4, 0.1]) # A
assert np.allclose(grads['B'], [0.0, 0.1]) # B 
assert np.allclose(grads['C'], [0.1, 0.4]) # C

prob_ev, probs = bn.marginal_probs(dag, params, evidence)
assert prob_ev==0.1
assert np.allclose(probs['B'], [0.0, 1.0])


############
"""
# Now compare with  https://github.com/pgmpy/pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianModel([('A','B'), ('A','C')])

# Defining individual CPDs.
cpd_A = TabularCPD(variable='A', variable_card=2, values=[thetaA])
cpd_B = TabularCPD(variable='B', variable_card=2, 
                   values=thetaB, evidence=['A'], evidence_card=[2])
cpd_C = TabularCPD(variable='C', variable_card=2, 
                   values=thetaC, evidence=['A'], evidence_card=[2])

model.add_cpds(cpd_A, cpd_B, cpd_C)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
model.check_model()

from pgmpy.inference import VariableElimination
infer = VariableElimination(model)

# cannot infer nodes that are observed
factorB = infer.query(['B'], evidence={'A': 1, 'C': 0}) ['B']
assert np.allclose(probs[1], factorB.values)
"""


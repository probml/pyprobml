"""
Inference in student network using autodiff compared to pgmpy.

Network is from
https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/2.%20Bayesian%20Networks.ipynb

I     D
| \   /
v   v
S   G
    |
    v
    L
    
All nodes are binary except G (grade), which has 3 levels.
"""
    
import superimport

import numpy as np 


import bayesnet_inf_autodiff as bn

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


dag = {'I':[], 'D':[], 'G':['I','D'], 'S':['I'], 'L':['G']}
paramsD = np.array([0.6, 0.4])
paramsI = np.array([0.7, 0.3])
paramsG = np.array([[0.3, 0.05, 0.9,  0.5],
                      [0.4, 0.25, 0.08, 0.3],
                      [0.3, 0.7,  0.02, 0.2]])
paramsG = np.reshape(paramsG, (3,2,2)) # paramsG[G,I,D]
paramsL = np.array([[0.1, 0.4, 0.99],
                    [0.9, 0.6, 0.01]])
paramsS = np.array([[0.95, 0.2],
                    [0.05, 0.8]])
params = {'D': paramsD, 'I': paramsI, 'G': paramsG, 'L': paramsL, 'S': paramsS}

inf_engine_ad = bn.BayesNetInfAutoDiff(dag, params)

def infer_autodiff(evidence, query):
    marginals = inf_engine_ad.infer_marginals(evidence)
    return marginals[query]

## Now compute results using pgmpy
    
# DAG is specified a list of pairs 
# eg [('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')]
edges = []
nodes = list(dag.keys())
for n in nodes:
    for pa in dag[n]:
        edge = (pa, n)
        edges.append(edge)
model = BayesianModel(edges)


cpd_d = TabularCPD(variable='D', variable_card=2, values=np.reshape(paramsD, (2, 1)))

cpd_i = TabularCPD(variable='I', variable_card=2, values=np.reshape(paramsI, (2, 1)))

cpd_g = TabularCPD(variable='G', variable_card=3, 
                   values=np.reshape(paramsG, (3, 2*2)),# flat 2d matrix
                   evidence=['I', 'D'],
                   evidence_card=[2,2])

cpd_l = TabularCPD(variable='L', variable_card=2, 
                   values=paramsL,
                   evidence=['G'],
                   evidence_card=[3])

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=paramsS,
                   evidence=['I'],
                   evidence_card=[2])


model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
model.check_model()
inf_engine_ve = VariableElimination(model) # compute elim order only once

def infer_pgmpy(evidence, query):
    factor = inf_engine_ve.query([query], evidence=evidence,joint=False) [query]    
    marginal = factor.values # convert from DiscreteFactor to np array
    return marginal

## Check both inference engines give same posterior marginals 
    
evlist = []
evlist.append({})
evlist.append({'G': 0, 'D': 0})
evlist.append({'L': 0, 'D': 1, 'S': 1})
for evidence in evlist:
    all_nodes = set(dag.keys())
    vis_nodes = set(evidence.keys())
    hid_nodes = all_nodes.difference(vis_nodes)
    for query in hid_nodes:
        prob_ad = infer_autodiff(evidence, query)        
        prob_pgm = infer_pgmpy(evidence, query)
        assert np.allclose(prob_ad, prob_pgm)

    
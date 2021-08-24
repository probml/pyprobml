# Based on https://github.com/probml/pmtk3/blob/master/demos/asiaDgm.m
# and https://github.com/pgmpy/pgmpy/blob/dev/examples/Inference%20in%20Bayesian%20Networks.ipynb

# Installation instructions
# https://github.com/pgmpy/pgmpy#installation

# Fetching the network
#!wget http://www.bnlearn.com/bnrepository/asia/asia.bif.gz
#!gzip -qd asia.bif.gz | rm asia.bif.gz

import superimport

from pgmpy.readwrite import BIFReader
reader = BIFReader('data/asia.bif')
asia_model = reader.get_model()

asia_model.nodes()

asia_model.edges()

CPDs = asia_model.get_cpds()

# Doing exact inference using Variable Elimination
from pgmpy.inference import VariableElimination
asia_infer = VariableElimination(asia_model)

# Computing the probability of bronc given smoke.
q = asia_infer.query(variables=['bronc'], evidence={'smoke': 0})
print(q['bronc'])

'''
Sanity check.
 p(A=t|T=t) = p(A=t) p(T=t|A=t) / [
    p(A=t) p(T=t|A=t)  + p(A=f) p(T=t|A=f)]
= 0.01 * 0.05 / (0.01 * 0.05 + 0.99 * 0.01)
= 0.0481
'''
# 0 = True. 1 = False
q = asia_infer.query(variables=['asia'], evidence={'tub': 0})
print(q['asia'])
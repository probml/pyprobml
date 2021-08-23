# Compute conditional independencies from a directed graphical model

# Uses this library
# https://github.com/ijmbarr/causalgraphicalmodels

# Code is based on
# https://fehiepsi.github.io/rethinking-numpyro/06-the-haunted-dag-and-the-causal-terror.html

import superimport
from causalgraphicalmodels import CausalGraphicalModel

dag = CausalGraphicalModel(
    nodes=["X", "Y", "C", "U", "B", "A"],
    edges=[
        ("X", "Y"),
        ("U", "X"),
        ("A", "U"),
        ("A", "C"),
        ("C", "Y"),
        ("U", "B"),
        ("C", "B"),
    ],
)


        
all_independencies = dag.get_all_independence_relationships()
print(all_independencies)
print('\n')


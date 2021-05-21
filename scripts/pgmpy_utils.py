# utility functions for pgmpy library

import pgmpy
import numpy as np
from graphviz import Digraph

def get_state_names(model):
  state_names = dict()
  for cpd in model.get_cpds():
    for k, v in cpd.state_names.items():
      state_names[k] = v
  return state_names

def get_marginals(model, evidence={}, inference_engine=None):
  if inference_engine is None:
    inference_engine = pgmpy.inference.VariableElimination(model) # more efficient to precompute this
  nodes = model.nodes()
  num_nodes = len(nodes)
  state_names = get_state_names(model)
  marginals = dict()
  for n in nodes:
    if n in evidence: # observed nodes
      v = evidence[n]
      if type(v) == str:
        v_ndx = state_names[n].index(v)
      else:
        v_ndx  = v
      nstates = model.get_cardinality(n)
      marginals[n] = np.zeros(nstates)
      marginals[n][v_ndx] = 1.0 # delta function on observed value
    else:
      probs = inference_engine.query([n], evidence=evidence).values
      marginals[n] = probs
  return marginals

def visualize_marginals(marginals, model):
    h = Digraph('pgm')
    for node_name, probs in marginals.items():
        yes_prob = probs[0]
        no_prob = probs[1]
        h.node(node_name, label='''<<TABLE>
      <TR PORT="header">
          <TD  COLSPAN="2"> {} </TD>
      </TR>
      <TR>
        <TD>yes</TD>
        <TD>{:.2f}</TD>
      </TR>
      <TR>
        <TD>no</TD>
        <TD>{:.2f}</TD>
      </TR>
    </TABLE>>'''.format(node_name, yes_prob, no_prob))

    edges = (model.edges())
    for item in edges:
        edge = list(item)
        h.edge(edge[0], edge[1])

    return h

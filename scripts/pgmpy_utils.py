# utility functions for pgmpy library
# authors: murphyk@, Drishttii@

#!pip install pgmpy
#!pip install graphviz

import superimport

import pgmpy
import numpy as np
import itertools
from graphviz import Digraph

def get_state_names(model, name):
  state_names = dict()
  cpd = model.get_cpds(name)
  state_names = cpd.state_names        
  return state_names

def get_all_state_names(model):
  state_names = dict()
  for cpd in model.get_cpds():
    for k, v in cpd.state_names.items():
      state_names[k] = v
  return state_names

def get_lengths(state_names, name):
  row = []
  for k, v in state_names.items():
    if (k==name):
      col = len(v)
    else:
      row.append(len(v))
  return row, col

def get_all_perms(states, name):
  all_list = []         
  for k, v in states.items():
    if (k==name):
      continue
    else:
      all_list.append(states[k])
  res = list(itertools.product(*all_list))
  resu = []
  for j in res:
    j = str(j)
    j = j.replace('(', '')
    j = j.replace(',', '')
    j = j.replace(')', '')
    j = j.replace("'", '')
    j = j.replace(' ', ', ')
    resu.append(j)    
  return resu

def visualize_model(model):
  h = Digraph('model_')
  # Adding each node
  for cpd in model.get_cpds():
    name = cpd.variable
    states = get_state_names(model, name)
    cpd = model.get_cpds(name)
    values = cpd.values
    values = values.T
    the_string = ""

    if len(states) == 1:
      rows = len(states[name]) 
      cols = rows
      row_string = ""
      for row in range(rows):
        col_string = ""
        for col in range(cols):
          if (row==0):
            inp = states[name]
            col_string = col_string + "<TD>" + str(inp[col]) + "</TD>" 
          else:
            two_dec = format(values[col], ".2f")
            col_string = col_string + "<TD>" + str(two_dec) + "</TD>" 

        row_string = row_string + "<TR>" + str(col_string) + "</TR>" 

    else:
      #lis = get_list(states, name)
      res = get_all_perms(states, name)
      r, c = get_lengths(states, name)
      rows = np.prod(r) + 1
      cols = c + 1
      values = values.reshape(rows-1, cols-1)
      row_string = ""
      for row in range(rows):
        col_string = ""
        if (row==0):
          for col in range(cols):
            if (col==0):
              col_string = col_string + "<TD>" + " " + "</TD>" 
            else:
              inp = states[name]
              col_string = col_string + "<TD>" + str(inp[col-1]) + "</TD>" 
          
          row_string = row_string + "<TR>" + str(col_string) + "</TR>" 

        else:
          for col in range(cols):
            if (col==0):
              col_string = col_string + "<TD>" + str(res[row-1]) + "</TD>"   
            else:
              two_dec = format(values[row-1][col-1], ".2f")
              col_string = col_string + "<TD>" + str(two_dec) + "</TD>" 
          
          row_string = row_string + "<TR>" + str(col_string) + "</TR>" 

    h.node(name, label = '''<<TABLE> 
    <TR PORT="header">
          <TD  COLSPAN="{}"> {} </TD>
      </TR> {} </TABLE>>'''.format(rows ,name, row_string))

  edges = (model.edges())
  for item in edges:
    edge = list(item)
    h.edge(edge[0], edge[1])

  return h

def get_marginals(model, evidence={}, inference_engine=None):
  if inference_engine is None:
    inference_engine = pgmpy.inference.VariableElimination(model) # more efficient to precompute this
  nodes = model.nodes()
  num_nodes = len(nodes)
  state_names = get_all_state_names(model)
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

def visualize_marginals(model, evidence, marginals):
    h = Digraph('pgm')
    for node_name, probs in marginals.items():
        states = get_state_names(model, node_name)
        rows = 2 #len(probs)
        cols = len(states[node_name])
        row_string = ""
        for row in range(rows):
          col_string = ""
          for col in range(cols):
            if (row==0):
              inp = states[node_name]
              col_string = col_string + "<TD>" + str(inp[col]) + "</TD>" 
            else:
              inp = round(probs[col], 2)
              col_string = col_string + "<TD>" + str(inp) + "</TD>" 

          row_string = row_string + "<TR>" + str(col_string) + "</TR>" 

        if node_name in evidence.keys():
          h.node(node_name, label = '''<<TABLE> 
            <TR PORT="header">
                <TD  BGCOLOR="#BEBEBE" COLSPAN="{}"> {} </TD>
            </TR> {} </TABLE>>'''.format(cols , node_name, row_string))
        else:
          h.node(node_name, label = '''<<TABLE> 
            <TR PORT="header">
                <TD  COLSPAN="{}"> {} </TD>
            </TR> {} </TABLE>>'''.format(cols , node_name, row_string))

    edges = (model.edges())
    for item in edges:
      edge = list(item)
      h.edge(edge[0], edge[1])

    return h


#!pip install pgmpy

# utility functions for pgmpy library
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

def get_values(model, name):
  cpd = model.get_cpds(name)
  values = cpd.values  
  return values

def get_column_string(input):
  string = "<TD>" + str(input) + "</TD>"
  return string

def get_row_string(input):
  string = "<TR>" + str(input) + "</TR>"
  return string

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
    j = j.replace(',', '')
    j = j.replace("'", '')
    resu.append(j)    
  return resu

def visualize_model(model):
  h = Digraph('model_')
  # Adding each node
  for cpd in model.get_cpds():
    name = cpd.variable
    states = get_state_names(model, name)
    values = get_values(model, name)
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
            col_string = col_string + get_column_string(inp[col])
          else:
            col_string = col_string + get_column_string(values[col])

        row_string = row_string + get_row_string(col_string)

    else:
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
              col_string = col_string + get_column_string(" ")
            else:
              inp = states[name]
              col_string = col_string + get_column_string(inp[col-1])
          
          row_string = row_string + get_row_string(col_string)

        else:
          for col in range(cols):
            if (col==0):
              col_string = col_string + get_column_string(res[row-1])  
            else:
              col_string = col_string + get_column_string(values[row-1][col-1])
          
          row_string = row_string + get_row_string(col_string)

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

def visualize_marginals(marginals, model):
    h = Digraph('pgm')
    for node_name, probs in marginals.items():
        states = get_state_names(model, node_name)
        rows = len(probs)
        cols = len(states[node_name])
        row_string = ""
        for row in range(rows):
          col_string = ""
          for col in range(cols):
            if (col==0):
              inp = states[node_name]
              col_string = col_string + get_column_string(inp[row])
            else:
              inp = round(probs[row], 2)
              col_string = col_string + get_column_string(inp)

          row_string = row_string + get_row_string(col_string)

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


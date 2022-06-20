'''
Draws a 6-node "tiny web" . This is from Section 2.11 of "Numerical Computing with MATLAB"
by Cleve Moler, SIAM, 2004.

This demo only operates on a single graph which consists of 6 nodes and draws
the initial state of the graph.
Furthermore, note that this demo does not show the PageRank is actually computed  Instead
the eigenvalue problem A*x=x is solved for x, where A is the Markov
transition matrix, A = p*G*D + e*z', where G is the binary matrix used here.
The method here is a simplistic random-hopping demonstration of the Markov
process, to motivate the A*x=x formulation of the problem. In this example,
A does control how the transitions are made, but the matrix A is not formed
explicitly.

Modifed by Kevin Murphy 26 Nov 97: I just changed the node names
to numbers, for brevity and ease of comparison to entries in the matrix/vector

Converted to Python, 07 June 2020.

Author : Kevin P. Murphy, Cleve Moler, Aleyna Kara

This file is based on https://github.com/probml/pmtk3/blob/master/demos/pagerankDemo.m
'''

import superimport

import numpy as np
from IPython.display import display
from graphviz import Digraph

# Plots the Directed Acylic Graph using graphviz
class DagPlotter:
  def __init__(self, G):
    self.G = G
    self.dot = Digraph(format='pdf')
    self.plot_nodes()
    self.plot_edges()

  def create_node(self, i):
    color = 'green' #'blue' if i==0 else 'green'
    self.dot.node(str(i+1), f'<X<SUB>{i+1}</SUB>>', fillcolor=color, style='filled', font_color='white')

  def plot_nodes(self):
    for i in range(self.G.shape[0]):
      self.create_node(i)

  def plot_edges(self):
    i, j = np.where(self.G==1) # adjacent nodes
    self.dot.edges(['{}{}'.format(j[k]+1, i[k]+1) for k in range(i.size)])


# Link structure of small web
G = np.array([[0, 0, 0, 1, 0, 1],[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0],[0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
graph = DagPlotter(G) # plots directed acyclic graph
graph.dot.render('../figures/pagerank-small-web')
display(graph.dot)
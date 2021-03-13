# simple regression tree on two inputs
# Author: Animesh Gupta 


import numpy as np
from numpy import ix_
import plotly.graph_objects as go 
from plotly.offline import plot


x11 = 5
x12 = 3
x21 = 7
x22 = 3

# Values for the tree in each region
r = np.arange(2,12,2)

# Create a mesh for X1 and X2
h = 0.1
M = 10.1

X1 = np.arange(0,M,h)
X2 = np.arange(0,M,h)

#The matrix to store the values of the tree
s = (len(X1),len(X2))
tree = np.zeros(s)

tree[ix_(X1 <= x11, X2 <= x21)] = r[0]
tree[ix_(X1 > x11, X2 <= x22)] = r[1]
tree[ix_(X1 > x11, X2 > x22)] = r[2]
tree[ix_(X1 <= min(x11,x22), X2 > x21)] = r[3]
tree[ix_(np.logical_and(X1 <= x11, X1 > x12), X2 > x21)] = r[4]


fig = go.Figure(data=[go.Mesh3d( 
  x=X1, y=X2, z=tree, color="seagreen", alphahull=4)]) 

plot(fig)

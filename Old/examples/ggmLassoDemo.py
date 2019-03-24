# Based on https://github.com/probml/pmtk3/blob/master/demos/ggmLassoDemo.m

import numpy as np
from sklearn.covariance import GraphLasso
import matplotlib.pyplot as plt
import networkx as nx
import os

#These column names are from 
#https://web.stanford.edu/~hastie/ElemStatLearn/datasets/sachs.info.txt
ProteinNames = ['praf', 'pmek', 'plcg', 'PIP2', 'PIP3','p44/42', 'pakts473', 'PKA', 'PKC', 'P38', 'pjnk']
NCols = len(ProteinNames)

#Import the data and convert to a numpy array
X = open(os.path.join('data', 'sachsCtsHTF.txt'), 'r').read().split()
X = [float(x) for x in X]
X = np.array(X).reshape(-1,NCols)
X -= X.mean(axis=0).reshape(1,-1)
X /= np.sqrt(1000) #same as http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/sachs.info

#Regularization parameters
Lambs = [36, 27, 7, 0] 

for lam in Lambs:
    GL = GraphLasso(lam)
    GL.fit(X)
    
    prec = GL.precision_
    
    #Form graph
    G = nx.Graph()
    G.add_nodes_from(ProteinNames)
    
    for i in range(NCols):
        for j in range(i):
            if prec[i,j]!=0:
                G.add_edges_from([(ProteinNames[i],ProteinNames[j])])
    
    ttl = 'lambda {}, nedges {}'.format(lam, len(G.edges))
    print(ttl)
    #plt.title(ttl)
    nx.draw_circular(G, edge_color ='blue', node_color ='yellow', with_labels = True, node_size = 3000)
    plt.savefig(os.path.join('figures', 'glassoSachsPython%s.pdf' % lam))
    #plt.savefig(os.path.join('figures', 'glassoSachs%s.pdf' % lam))
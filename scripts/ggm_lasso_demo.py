# Applying the graphical lasso to the flow-cytometry dataset 
# Author: Drishtii@
# Based on: https://github.com/probml/pmtk3/blob/master/demos/ggmLassoDemo.m 
# Sourced from: https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/examples/Protein%20Flow%20Cytometry.ipynb

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import networkx as nx
import pyprobml_utils as pml

url = 'https://raw.githubusercontent.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/master/data/protein.data'
df = pd.read_csv(url, header=None, sep=' ')

X = df.to_numpy()

protein_names = ['Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38', 'Jnk']
p = len(protein_names)

# the empirical covariance matrix
S = np.cov(X, rowvar=False)/1000
lambdas = [36, 27, 7]
theta_estimates = []

#  In practice it is informative to examine the different sets of graphs that are obtained as λ is varied. Figure shows 4 different
#  solutions. The graph becomes more sparse as the penalty parameter is increased.

for lam in lambdas:
    # theta should be symmetric positive-definite
    theta = cp.Variable(shape=(p, p), PSD=True)
    # An alternative formulation of the problem () can be posed,
    # where we don't penalize the diagonal of theta.
    l1_penalty = sum([cp.abs(theta[i, j])
                      for i in range(p)
                      for j in range(p) if i != j])
    objective = cp.Maximize(
        cp.log_det(theta) - cp.trace(theta@S) - lam*l1_penalty)
    problem = cp.Problem(objective)
    problem.solve()
    if problem.status != cp.OPTIMAL:
        raise Exception('CVXPY Error')
    theta_estimates.append(theta.value)

lambdas.append(0)
theta_estimates.append(np.linalg.inv(S))

# Four different graphical-lasso solutions for the flow-cytometry data.
tmp = {name: name for name in protein_names}
#fig, axarr = plt.subplots(2, 2, figsize=(6, 6), dpi=150)
#plt.subplots_adjust(wspace=0.1, hspace=0.1)

angles = np.linspace(0, 1, p + 1)[:-1] * 2 * np.pi + np.pi/2
for plot_idx in range(4):
    cons = np.argwhere(np.abs(theta_estimates[plot_idx]) > 0.00001)
    G, node_pos = nx.Graph(), {}
    for i, node in enumerate(protein_names):
        G.add_node(node)
        node_pos[node] = np.array([np.cos(angles[i]), np.sin(angles[i])])
    for i in range(cons.shape[0]):
        G.add_edge(protein_names[cons[i, 0]], protein_names[cons[i, 1]])
    #ax = axarr[plot_idx//2, plot_idx % 2]
    fig, ax = plt.subplots()
    nx.draw(G, node_pos, node_size=3, with_labels=False, ax=ax,
            edge_color='#174A7E', width=0.6, node_color='#174A7E')
    description = nx.draw_networkx_labels(G, node_pos, labels=tmp, ax=ax)
    for (i, (node, t)) in enumerate(description.items()):
        t.set_position((np.cos(angles[i]), np.sin(angles[i])+0.08))
        t.set_fontsize(7)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.text(0, 1.18, f'λ = {lambdas[plot_idx]}', fontsize=8)
    plt.tight_layout()
    pml.savefig(f'ggm_lasso{plot_idx}.pdf')
plt.show()

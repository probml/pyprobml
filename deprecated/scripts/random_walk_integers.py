import superimport

import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt

t = np.full((1, 20), 0.5).reshape(-1, )
T = np.diag(t, -1) + np.diag(t, 1)
T[0, 0] = 0.5
T[20, 20] = 0.5

plotPoints = np.array([0, 1, 2, 3, 10, 100, 200, 400])
p0_a = np.zeros((1,21)).reshape(-1, 1)
p0_a[10] = 1
p0_b = np.zeros((1,21)).reshape(-1, 1)
p0_b[17] = 1

p0 = {}
p0[0] = p0_a
p0[1] = p0_b

x = [0, 5, 10, 15, 20]
init = [10, 17]

for i in range(2):
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10,10))
    plt.setp(axes, xticks=x) 
    fig.suptitle('Initial condition X0 = {}'.format(init[i]))
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-0.5, 20.5)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel('p ({}) x'.format(plotPoints[0]))
    axes[0].stem(p0[i], markerfmt='o', linefmt='black')
    for j in range(1, len(plotPoints)): 
        N = plotPoints[j]
        w = np.linalg.matrix_power(T, N)
        q = np.matmul(w, p0[i])  
        axes[j].set_aspect('equal')
        axes[j].set_xlim(-0.5, 20.5)
        axes[j].set_ylim(0, 1.2)
        axes[j].set_ylabel('p ({}) x'.format(plotPoints[j]))
        q = q*(j)
        axes[j].stem(q, markerfmt='o', linefmt='black')
        plt.tight_layout()
    pml.savefig('Initial_state_'+str(init[i])+'.pdf')
    plt.savefig('Initial_state_'+str(init[i])+'.pdf')
    plt.show()
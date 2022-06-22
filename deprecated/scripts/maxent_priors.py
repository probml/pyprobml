

# jeffreys prior for bernoulli using 2 paramterizatiobs
# fig 1.10 of 'Bayeysian Modeling and Computation'

import superimport

import numpy as np
import matplotlib.pyplot as plt 
import pyprobml_utils as pml


from scipy import stats
from scipy.stats import entropy
from scipy.optimize import minimize

C = 10
xs  = np.arange(1,C+1)

cons = [[{"type": "eq", "fun": lambda x: np.sum(x) - 1}],
        [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
         {"type": "eq", "fun": lambda x: 1.5 - np.sum(x *xs)}],
        [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
         {"type": "eq", "fun": lambda x: np.sum(x[[2, 3]]) - 0.8}]]

max_ent = []
names= ['unconstrained', 'mean of 1.5', 'p(3,4)=0.8']
for i, c in enumerate(cons):
    val = minimize(lambda x: -entropy(x), 
                   x0=[1/C]*C, 
                   bounds=[(0., 1.)] * C,
                   constraints=c)['x']
    max_ent.append(entropy(val))
    plt.plot(xs, val, 'o--', lw=2.5, label=names[i])
    #plt.stem(xs, val, label=names[i])
plt.xlabel(r"$\theta$")
plt.ylabel(r"$p(\theta)$")
plt.legend()

pml.savefig("maxent_priors.pdf", dpi=300)
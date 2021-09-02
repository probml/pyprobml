

# jeffreys prior for bernoulli using 2 paramterizatiobs
# fig 1.9 of 'Bayeysian Modeling and Computation'

import superimport

import numpy as np
import matplotlib.pyplot as plt 
import pyprobml_utils as pml


from scipy import stats


x = np.linspace(0, 1, 500)
params = [(0.5, 0.5), (1, 1), (3,3), (100, 25)]

labels = ["Jeffreys", "MaxEnt", "Weakly  Informative",
          "Informative"]

_, ax = plt.subplots()
for (α, β), label, c in zip(params, labels, (0, 1, 4, 2)):
    pdf = stats.beta.pdf(x, α, β)
    ax.plot(x, pdf, label=f"{label}", c=f"C{c}", lw=3)
    ax.set(yticks=[], xlabel="θ", title="Priors")
    ax.legend()
pml.savefig("binomial_priors.pdf", dpi=300)

import numpy as np
import matplotlib.pyplot as plt
import os

# Bar graphs showing a uniform discrete distribution and another with full prob on one value.

X = np.arange(1, 5)
UniProbs = np.repeat(1.0/len(X),len(X))

def MakeG(Probs,SaveN):
    fig, ax = plt.subplots()
    ax.bar(X, Probs, align='center')
    plt.xlim([min(X) - .5, max(X) + .5])
    plt.xticks(X)
    plt.yticks(np.linspace(0, 1, 5))
    #plt.savefig(SaveN)
    plt.savefig(os.path.join('figures', SaveN))

MakeG(UniProbs, "ProbsDiscreteUniform")
MakeG([1, 0, 0, 0], "ProbsDiscreteDelta")

plt.show()

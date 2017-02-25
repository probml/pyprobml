import numpy as np
import matplotlib.pyplot as plt
import os

# Bar graphs showing a uniform discrete distribution and another with full prob on one value.
FIGURE_DIR = 'figures'

X = np.arange(1, 5)
UniProbs = np.repeat(1.0/len(X),len(X))

def MakeG(Probs,SaveN):
    fig, ax = plt.subplots()
    ax.bar(X, Probs, align='center')
    plt.xlim([min(X) - .5, max(X) + .5])
    plt.xticks(X)
    plt.yticks(np.linspace(0, 1, 5))
    if not os.path.exists(FIGURE_DIR):
        os.mkdir(FIGURE_DIR)
    figure_path = os.path.join(FIGURE_DIR, SaveN)
    plt.savefig(figure_path)

MakeG(UniProbs, "ProbsDiscreteUniform")
MakeG([1, 0, 0, 0], "ProbsDiscreteDelta")

plt.show()

# Bar graphs showing a uniform discrete distribution and another with full prob on one value.

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml


X = np.arange(1, 5)
UniProbs = np.repeat(1.0/len(X),len(X))

def MakeG(Probs,SaveN):
    fig, ax = plt.subplots()
    ax.bar(X, Probs, align='center')
    plt.xlim([min(X) - .5, max(X) + .5])
    plt.xticks(X)
    plt.yticks(np.linspace(0, 1, 5))
    pml.savefig(SaveN)

#MakeG(UniProbs, "unifHist.pdf")
#MakeG([1, 0, 0, 0], "deltaHist.pdf")

MakeG(UniProbs, "uniform_histogram.pdf")
MakeG([1, 0, 0, 0], "delta_histogram.pdf")

plt.show()

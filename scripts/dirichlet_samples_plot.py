

import numpy as np
import matplotlib.pyplot as plt
import os

def save_fig(fname):
    figdir = os.path.join(os.environ["PYPROBML"], "figures")
    plt.tight_layout()    
    fullname = os.path.join(figdir, fname)
    print('saving to {}'.format(fullname))
    plt.savefig(fullname)

np.random.seed(0)

NSamples = 5
X = np.arange(1, 6)

# This function generators a few bar graphs showing samples generated from a
# Dirichlet distribution with a param vector with elements = alpha.
def MakeDirSampleFig(alpha):
    AlphaVec = np.repeat(alpha, NSamples)
    samps = np.random.dirichlet(AlphaVec, NSamples)
    fig, ax = plt.subplots(NSamples)
    fig.suptitle('Samples from Dir (alpha=' + str(alpha) +')', y=1)
    fig.tight_layout()

    for i in range(NSamples):
        ax[i].bar(X, samps[i, :], align='center')
        ax[i].set_ylim([0, 1])
        ax[i].yaxis.set_ticks([0, .5, 1])
        ax[i].set_xlim([min(X) - .5, max(X) + .5])

    plt.draw()
    SaveN = "dirSample" + str(int(np.round(10*alpha))) + ".pdf"
    save_fig(SaveN)

MakeDirSampleFig(.1)
MakeDirSampleFig(1.0)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import os

def save_fig(fname):
    figdir = os.path.join(os.environ["PYPROBML"], "figures")
    plt.tight_layout()    
    fullname = os.path.join(figdir, fname)
    print('saving to {}'.format(fullname))
    plt.savefig(fullname)

from scipy.stats import dirichlet

#Points where we evaluate the pdf
x = np.linspace(0.001, .999, 100)

#Given an alpha parameter, this returns a pdf function
def MakeBeta(alpha):
    def Beta(y):
        return dirichlet.pdf([y, 1 - y], alpha)
    Beta = np.vectorize(Beta)
    return Beta

#Makes strings for the legend:
def MakeLabel(Data,which):
    alpha = Data[which]
    lab = which + " Be(" + str(alpha[0]) + ", " + str(alpha[1]) + ")"
    return lab

#Forms graph give the parameters of the prior, likelihood and posterior:
def MakeGraph(Data,SaveName):
    prior = MakeBeta(Data['prior'])(x)
    likelihood = MakeBeta(Data['lik'])(x)
    posterior = MakeBeta(Data['post'])(x)

    fig, ax = plt.subplots()
    ax.plot(x, prior, 'r', label=MakeLabel(Data, "prior"), linewidth=2.0)
    ax.plot(x, likelihood, 'k--', label=MakeLabel(Data, "lik"), linewidth=2.0)
    ax.plot(x, posterior, 'b--', label=MakeLabel(Data, "post"), linewidth=2.0)
    ax.legend(loc='upper left', shadow=True)
    save_fig(SaveName)
    plt.show()

Data1 = {'prior': [1, 1],
       'lik': [5, 2],
       'post': [5, 2]}

Data2 = {'prior': [1, 1],
       'lik': [41, 11],
       'post': [41, 11]}

Data3 = {'prior': [2, 2],
       'lik': [5, 2],
       'post': [6, 3]}

Data4 = {'prior': [2, 2],
       'lik': [41, 11],
       'post': [42, 12]}

MakeGraph(Data1, "betaPost1")
MakeGraph(Data2, "betaPost2")
MakeGraph(Data3, "betaPost3")
MakeGraph(Data4, "betaPost4")

plt.show()

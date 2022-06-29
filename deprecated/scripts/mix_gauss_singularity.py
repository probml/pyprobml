# Ilustration of how singularities can arise in the likelihood function
# of GMMs

# Author: Gerardo Durán Martín

import superimport

import numpy as np
import pyprobml_utils as pml
from scipy.stats import norm
import matplotlib.pyplot as plt

def main():
    f1 = norm(loc=0.5, scale=0.12)
    f2 = norm(loc=0.15, scale=0.02)

    domain = np.arange(0, 1, 0.001)
    datapoints = np.array([0.15, 0.21, 0.25, 0.32, 0.45, 0.58, 0.72, 0.88])

    def f3(x): return f1.pdf(x) + f2.pdf(x)

    plt.stem(datapoints, f3(datapoints), linefmt="tab:green", basefmt=" ")
    for datapoint in datapoints:
        plt.scatter(datapoint, -0.1, c="black", zorder=2)
    plt.plot(domain, f3(domain), c="tab:red", zorder=1)
    plt.xlabel("x", fontsize=13)
    plt.ylabel("p(x)", fontsize=13)
    plt.xlim(0.01, 0.98)
    plt.xticks([])
    plt.yticks([])
    pml.savefig('gmm_singularity.pdf')
    plt.show()


if __name__ == "__main__":
    main()

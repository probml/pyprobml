# Example of an Extended Kalman Filter using
# a figure-8 nonlinear dynamical system.
# For futher reference and examples see:
#   * Section on EKFs in PML vol2 book
#   * https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb
#   * Nonlinear Dynamics and Chaos - Steven Strogatz
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import ekf_continuous_demo
import matplotlib.pyplot as plt

figures = ekf_continuous_demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

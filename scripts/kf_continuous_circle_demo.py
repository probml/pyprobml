# Example of a Kalman Filter procedure
# on a continuous system with imaginary eigenvalues
# and discrete samples
# For futher reference and examples see:
#   * Section on Kalman Filters in PML vol2 book
#   * Nonlinear Dynamics and Chaos - Steven Strogatz
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import kf_continuous_circle_demo as demo
import matplotlib.pyplot as plt

figures = demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

# Example of a 1D pendulum problem applied to the Extended Kalman Filter,
# the Unscented Kalman Filter, and the Particle Filter (boostrap filter)
# Additionally, we test the particle filter when the observations have a 40%
# probability of being perturbed by a uniform(-2, 2) distribution
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import pendulum_1d_demo
import matplotlib.pyplot as plt

figures = pendulum_1d_demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

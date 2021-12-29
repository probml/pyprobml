# This demo exemplifies the use of the Kalman Filter
# algorithm when the linear dynamical system induced by the
# matrix A has imaginary eigenvalues
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import kf_spiral_demo as demo
import matplotlib.pyplot as plt

figures = demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

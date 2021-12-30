# Demo of the bootstrap filter under a
# nonlinear discrete system
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import bootstrap_filter_demo as demo
import matplotlib.pyplot as plt

figures = demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

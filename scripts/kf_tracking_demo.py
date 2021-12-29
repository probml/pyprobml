# This script produces an illustration of Kalman filtering and smoothing
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import kf_tracking_demo as demo
import matplotlib.pyplot as plt

figures = demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

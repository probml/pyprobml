# Parallel Kalman Filter demo: this script simulates
# 4 missiles as described in the section "state-space models".
# Each of the missiles is then filtered and smoothed in parallel
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import kf_parallel_demo as demo
import matplotlib.pyplot as plt

figures = demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

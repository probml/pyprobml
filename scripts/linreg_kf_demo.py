# Online Bayesian linear regression using Kalman Filter
# Based on: https://github.com/probml/pmtk3/blob/master/demos/linregOnlineDemoKalman.m
# Author: Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import linreg_kf_demo
import matplotlib.pyplot as plt

figures = linreg_kf_demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

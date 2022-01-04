# Online learning of a logistic
# regression model using the Exponential-family
# Extended Kalman Filter (EEKF) algorithm
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import eekf_logistic_regression_demo
import matplotlib.pyplot as plt

figures = eekf_logistic_regression_demo.main()
data = figures.pop("data")
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

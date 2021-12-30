# Compare extended Kalman filter with unscented kalman filter on a nonlinear 3d tracking problem
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import ekf_vs_ukf
import matplotlib.pyplot as plt

figures = ekf_vs_ukf.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

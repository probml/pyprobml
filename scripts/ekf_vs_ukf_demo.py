# Compare extended Kalman filter with unscented kalman filter on a nonlinear 3d tracking problem
# Author: Gerardo Duran-Martin (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import ekf_vs_ukf
import pyprobml_utils as pml

figures = ekf_vs_ukf.main()
for name, figure in figures.items():
    figure.show()
    pml.savefig(f"{name}.pdf")
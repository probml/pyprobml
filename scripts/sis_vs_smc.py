# We replicate Figures 2, 3 and 4 from the paper by
# Christian A. Naesseth, Fredrik Lindsten, Thomas B. Schön:
#   “Elements of Sequential Monte Carlo”, 2019; arXiv:1903.04797
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import sis_vs_smc_demo
import matplotlib.pyplot as plt

figures = sis_vs_smc_demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

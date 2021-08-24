import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from scipy.stats import norm

x = np.linspace(-3, 3, 100)
y = norm.pdf(x)
f = norm.cdf(x)

plt.figure()
plt.plot(x, f)
plt.title('CDF')
pml.savefig('gaussianCDF.pdf')
plt.show()

plt.figure()
plt.plot(x, y)
pml.savefig('gaussianPDF.pdf')
plt.show()

plt.figure()
plt.plot(x, y)
x_sep_left = norm.ppf(0.025)
x_sep_right = norm.ppf(0.975)
x_fill_left = np.linspace(-3, x_sep_left, 100)
x_fill_right = np.linspace(x_sep_right, 3, 100)
plt.fill_between(x_fill_left,
                norm.pdf(x_fill_left),
                color='b')
plt.fill_between(x_fill_right,
                norm.pdf(x_fill_right),
                color='b')
plt.annotate(r'$\alpha/2$', xy=(x_sep_left, norm.pdf(x_sep_left)),
            xytext=(-2.5, 0.1),
            arrowprops=dict(facecolor='k'))
plt.annotate(r'$1-\alpha/2$', xy=(x_sep_right, norm.pdf(x_sep_right)),
            xytext=(2.5, 0.1),
            arrowprops=dict(facecolor='k'))
plt.ylim([0, 0.5])
pml.savefig('gaussianQuantile.pdf')
plt.show()

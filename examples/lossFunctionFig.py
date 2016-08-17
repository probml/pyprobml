#!/usr/bin/env python3

# Plots loss functions of form |x|**q

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import t, laplace, norm

x = np.linspace(-4, 4, 100)
pl.title('|x|^0.2')
pl.plot(x, np.absolute(x)**.2)
pl.savefig('lossFunctionFig_01.png')

pl.figure()
pl.title('|x|')
pl.plot(x, np.absolute(x))
pl.savefig('lossFunctionFig_02.png')

pl.figure()
pl.title('|x|^2')
pl.plot(x, np.absolute(x)**2)
pl.savefig('lossFunctionFig_03.png')
pl.show()

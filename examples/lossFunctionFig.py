#!/usr/bin/env python3

# Plots loss functions of form |x|**q

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import t, laplace, norm
from utils.util import save_fig

x = np.linspace(-4, 4, 100)
pl.title('|x|^0.2')
pl.plot(x, np.absolute(x)**.2)
save_fig('lossFunctionFig_01.png')

pl.figure()
pl.title('|x|')
pl.plot(x, np.absolute(x))
save_fig('lossFunctionFig_02.png')

pl.figure()
pl.title('|x|^2')
pl.plot(x, np.absolute(x)**2)
save_fig('lossFunctionFig_03.png')
pl.show()

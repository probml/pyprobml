
#https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html#
import superimport

import pygam

from pygam.datasets import wage

X, y = wage()

from pygam import LinearGAM, s, f

#Let’s fit a spline term to the first 2 features, and a factor term to the 3rd feature.

gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)

gam.summary()
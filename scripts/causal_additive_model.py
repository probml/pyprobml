# Based on "Elements of causal inference" code snippet 4.14


#https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html#
import pygam
from pygam import LinearGAM, s

import numpy as np
np.random.seed(42)
N = 200
X = np.random.randn(N)
Y = np.power(X, 3) + np.random.randn(N)

gam_fwd = LinearGAM(s(0)).fit(X, Y)
Yhat = gam_fwd.predict(X)
residuals_fwd = Y - Yhat 
loglik_fwd = -(np.log(np.var(X)) + np.log(np.var(residuals_fwd)))
print(loglik_fwd)

gam_back = LinearGAM(s(0)).fit(Y, X)
Xhat = gam_fwd.predict(Y)
residuals_back = X - Xhat
loglik_back = -(np.log(np.var(Y)) + np.log(np.var(residuals_back)))
print(loglik_back)

# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

import time

import numpy as np

import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

rng = np.random.RandomState(0)

# Generate sample data
X = 15 * rng.rand(100, 1)
y = np.sin(X).ravel()
y += 3 * (0.5 - rng.rand(X.shape[0]))  # add noise

# Fit KernelRidge with parameter selection based on 5-fold cross validation
param_grid = {
    "alpha": [1e0, 1e-1, 1e-2, 1e-3],
    "kernel": [
        ExpSineSquared(l, p)
        for l in np.logspace(-2, 2, 10)
        for p in np.logspace(0, 2, 10)
    ],
}
kr = GridSearchCV(KernelRidge(), param_grid=param_grid)
stime = time.time()
kr.fit(X, y)
print("Time for KRR fitting: %.3f" % (time.time() - stime))

gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
gpr = GaussianProcessRegressor(kernel=gp_kernel)
stime = time.time()
gpr.fit(X, y)
print("Time for GPR fitting: %.3f" % (time.time() - stime))

# Predict using kernel ridge
X_plot = np.linspace(0, 20, 10000)[:, None]
stime = time.time()
y_kr = kr.predict(X_plot)
print("Time for KRR prediction: %.3f" % (time.time() - stime))

# Predict using gaussian process regressor
stime = time.time()
y_gpr = gpr.predict(X_plot, return_std=False)
print("Time for GPR prediction: %.3f" % (time.time() - stime))

stime = time.time()
y_gpr, y_std = gpr.predict(X_plot, return_std=True)
print("Time for GPR prediction with standard-deviation: %.3f" % (time.time() - stime))

# Plot results
plt.figure(figsize=(10, 5))
lw = 2
plt.scatter(X, y, c="k", label="data")
plt.plot(X_plot, np.sin(X_plot), color="navy", lw=lw, label="True")
#plt.plot(X_plot, y_kr, color="turquoise", lw=lw, label="KRR (%s)" % kr.best_params_)
plt.plot(X_plot, y_kr, color="turquoise", lw=lw, label="KRR")
#plt.plot(X_plot, y_gpr, color="darkorange", lw=lw, label="GPR (%s)" % gpr.kernel_)
plt.plot(X_plot, y_gpr, color="darkorange", lw=lw, label="GPR")
plt.fill_between(
    X_plot[:, 0], y_gpr - y_std, y_gpr + y_std, color="darkorange", alpha=0.2
)
plt.xlabel("data")
plt.ylabel("target")
plt.xlim(0, 20)
plt.ylim(-4, 4)
plt.title("GPR versus Kernel Ridge")
plt.legend(loc="best", scatterpoints=1, prop={"size": 8})
pml.savefig('krr_vs_gpr.pdf')
plt.show()
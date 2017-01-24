import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simple line graph of the density function of a standard normal.

X = np.linspace(-3, 3, 500)
density = norm.pdf(X, 0, 1)

fig, ax = plt.subplots()
ax.plot(X, density)
plt.title("Gaussian pdf")
plt.savefig("figures/Gaussianpdf")
plt.show()

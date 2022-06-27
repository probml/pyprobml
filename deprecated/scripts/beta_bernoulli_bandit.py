# Creates a figure for a beta distribution
# Author : Aleyna Kara (@karalleyna)

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

a, b = 2, 2
theta_idx = 1
colors = ["tab:blue", "tab:orange"]

x = np.linspace(beta.ppf(0, a, b),beta.ppf(1, a, b), 100)

plt.figure(figsize=(7,7))
plt.xlim(0, 1)
plt.xticks([1])
plt.yticks([1])
plt.plot(x, beta.pdf(x, a, b), '-', c=colors[theta_idx-1], linewidth=3)
plt.xlabel(f'$\Theta_{theta_idx}$', fontsize='15')
plt.ylabel(f'f($\Theta_{theta_idx}$)', fontsize='15')

plt.savefig(f"action_{theta_idx}_{a}{b}.png", bbox_inches='tight')
plt.show()
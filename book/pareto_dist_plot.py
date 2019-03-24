# Plots Pareto distribution

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


from scipy.stats import pareto

params = [(0.001, 1), (0.001, 2), (0.001, 3), (0.1, 2)]
styles = ['b-', 'r:', 'k-.', 'g--']
labels = ['m={:.2f}, k={:.2f}'.format(m, k) for m, k in params]
x = np.linspace(0, 0.5, 100)
  
for i, param in enumerate(params):
  m, k = param
  probabilities = pareto.pdf(x, k, scale=m)
  plt.plot(x, probabilities, styles[i], label=labels[i])

plt.title('Pareto Distribution')
plt.legend()
plt.axis((-0.1, 0.5, 0, 20))
save_fig('pareto-pdf.pdf')
plt.show()


for i, param in enumerate(params):
  m, k = param
  probabilities = pareto.pdf(x, k, scale=m)
  plt.loglog(x, probabilities, styles[i], label=labels[i])

plt.title('Log Pareto Distribution')
plt.legend()
save_fig('pareto-log-pdf2.pdf')
plt.show()
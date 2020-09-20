# Plots Pareto distribution

import numpy as np
import matplotlib.pyplot as plt
import os
#figdir = os.path.join(os.environ["PYPROBML"], "figures")
figdir = '../figures'
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


from scipy.stats import pareto

params = [(0.1, 1), (0.1, 2),  (0.2, 1), (0.2, 2)]
styles = ['b-', 'r:', 'k-.', 'g--']
labels = ['m={:.2f}, k={:.2f}'.format(m, k) for m, k in params]
x = np.linspace(0, 1, 1000)
  
for i, param in enumerate(params):
  m, k = param
  probabilities = pareto.pdf(x, k, scale=m)
  plt.plot(x, probabilities, styles[i], label=labels[i])

plt.title('Pareto Distribution')
plt.legend()
plt.axis((0.0, 0.5, 0, 20))
save_fig('paretoPdf.pdf')
plt.show()


for i, param in enumerate(params):
  m, k = param
  probabilities = pareto.pdf(x, k, scale=m)
  plt.loglog(x, probabilities, styles[i], label=labels[i])
  
plt.xlim(0.05, 1)
plt.title('Log Pareto Distribution')
plt.legend()
save_fig('paretoLogPdf.pdf')
plt.show()
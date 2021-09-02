# Generates from stick-breaking construction

import superimport

import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt

alphas = [2, 5]
nn = 20

# From MATLAB's random generator.
match_matlab = True  # Set True to match MATLAB's figure.
beta1 = [0.4428, 0.0078, 0.1398, 0.5018, 0.0320, 0.3614, 0.8655,
         0.6066, 0.2783, 0.4055, 0.1617, 0.3294, 0.0956, 0.1245,
         0.2214, 0.3461, 0.5673, 0.2649, 0.1153, 0.7366]
beta2 = [0.2037, 0.3486, 0.5342, 0.0609, 0.2997, 0.2542, 0.0860,
         0.1865, 0.0510, 0.4900, 0.4891, 0.7105, 0.7633, 0.1619,
         0.3604, 0.0604, 0.1312, 0.3338, 0.2036, 0.1306]
beta3 = [0.3273, 0.0253, 0.1415, 0.1574, 0.0460, 0.0721, 0.3386,
         0.1817, 0.2750, 0.0791, 0.0535, 0.1091, 0.1935, 0.0550,
         0.3977, 0.2322, 0.0270, 0.0871, 0.0144, 0.4171]
beta4 = [0.0395, 0.1170, 0.0272, 0.0155, 0.2190, 0.1812, 0.0569,
         0.2569, 0.1311, 0.0388, 0.3619, 0.1974, 0.3794, 0.1917,
         0.0670, 0.0294, 0.0957, 0.1267, 0.0381, 0.2525]
beta_all = [np.array(beta1), np.array(beta2), np.array(beta3), np.array(beta4)]

np.random.seed(0)
fig, axs = plt.subplots(2, 2)
fig.tight_layout()

for ii, alpha in enumerate(alphas):
    for trial in range(2):
        if match_matlab:
            beta = beta_all[ii*2+trial]
        else:
            beta = np.random.beta(1, alpha, [nn])
        neg = np.cumprod(1-beta)
        neg[1:] = neg[:-1]
        neg[0] = 1
        pi = beta*neg
        axs[ii, trial].bar(np.arange(nn), pi, edgecolor='k')
        axs[ii, trial].set_title(r'$\alpha = %s$' % alpha)

pml.savefig("stickBreakingDemo.pdf")
plt.show()

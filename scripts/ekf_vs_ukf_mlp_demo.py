# Demo showcasing the training of an MLP with a single hidden layer using 
# Extended Kalman Filtering (EKF) and Unscented Kalman Filtering (UKF).
# In this demo, we consider the latent state to be the weights of an MLP.
#   The observed state at time t is the output of the MLP as influenced by the weights
#   at time t-1 and the covariate x[t].
#   The transition function between latent states is the identity function.
# For more information, see
#   * Neural Network Training Using Unscented and Extended Kalman Filter
#       https://juniperpublishers.com/raej/RAEJ.MS.ID.555568.php
#   * UKF-based training algorithm for feed-forward neural networks with
#     application to XOR classification problem
#       https://ieeexplore.ieee.org/document/6234549
# Author: Gerardo Durán-Martín (@gerdm)

# !pip install git+git://github.com/probml/jsl

from jsl.demos import ekf_vs_ukf_mlp_demo
import matplotlib.pyplot as plt

figures = ekf_vs_ukf_mlp_demo.main()
for name, figure in figures.items():
    filename = f"./../figures/{name}.pdf"
    figure.savefig(filename)
plt.show()

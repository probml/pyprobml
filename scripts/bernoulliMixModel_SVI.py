# Author: Meduri Venkata Shivaditya
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bayespy.nodes import Categorical, Dirichlet, Beta, Mixture, Bernoulli
from bayespy.inference import VB
##### Bernoulli Mixture Model for MNIST dataset using Stochastic Variational Inference #####
# Figure number 21.19 in the book "Probabilistic Machine Learning: An Introduction by Kevin P. Murphy"
# External Dependencies : BayesPy(pip install bayespy)
"""
We’ll use stochastic variational inference to fit the mixture model. 
Briefly, this means that to go from the non-Bayesian model to a variational Bayesian model, 
we’ll replace each point parameter of our model with a probability distribution, called the “variational posterior”.
Then we’ll optimize the variables of those variational posterior distributions to be as close as possible to the true posteriors.

Variational Posterior for Binary image clusters is Beta distribution
Variational Posterior for cluster assignment probabilities is Dirichlet distribution

Traditional EM algorithm is costly especially due to the M step that focuses on maximizing 
the likelihood and sometimes gets stuck at the local maximum
Stochastic variational inferernce(SVI) is a more efficient process to estimate the parameters
It is extremely helpful in situations involving image data as they are generally massive

SVI Algorithm:
Each time SVI picks a sample, updates the corresponding local parameters, 
and computes the update of the global parameters as if all the m samples are identical to the picked sample.
Finally it incorporates this global parameter value into previous 
computations of the global parameters, by means of an exponential moving average.

Local Parameters :  Categorical distribution for the group assignments
Global Parameters : Group assignment probabilities an uninformative Dirichlet prior, Probability of pixel being 1 given by Beta Prior

For more information on implementation of Bernoulli Mixture Model using BayesPY 
visit https://www.bayespy.org/examples/bmm.html#results

"""
np.random.seed(1)
def mnist_data(n):
    #Downloading data from tensorflow datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #Converting images to binary
    x = (x_train>128).astype('int') #Coverting to binary
    x_train = x[np.random.randint(x.shape[0], size=n)]
    x_train = x_train.reshape((n, 784))
    return x_train
def plot_data(wts, mns):
    #Plotting the cluster means and their corresponding weights
    fig, ax = plt.subplots(4, 5)
    k = 0
    for i in range(4):
        for j in range(5):
            ax[i][j].imshow(mns[k].reshape(28, 28), cmap=plt.cm.gray)
            ax[i][j].set_title("%1.2f" % wts[k])
            ax[i][j].axis("off")
            k = k + 1
    fig.tight_layout(pad=1.0)
    plt.savefig("mixBernoulliMnist_SVI.png", dpi=300)
    plt.show()
def main():
    K = 20 # Number of mixture components
    D = 784 # Number of dimensions
    N = 60000 #Size of data
    data = mnist_data(N)
    subset_size = 10000 #Sample subset size for SVI
    # We give the group assignment probabilities an uninformative Dirichlet prior
    R = Dirichlet(K*[1e-5], name='R') #Dirichlet prior
    # We use the categorical distribution for the group assignments
    Z = Categorical(R, plates = (subset_size,1), name = 'Z')
    #Each group has a probability of a 1 for each pixel. These probabilities are given beta priors:
    P = Beta([0.5, 0.5], plates=(D,K), name='P') #Beta Prior
    #The pixel values of the MNIST images are modelled with the Bernoulli distribution:
    X = Mixture(Z, Bernoulli, P, name='X')
    Q = VB(Z, R, X, P) #Variational Bayesian Inference Engine
    delay = 1
    forgetting_rate = 0.7
    P.initialize_from_random()
    Q.ignore_bound_checks = True
    ##### SVI Algorithm #####
    for n in range(10):
      subset = np.random.choice(len(data), subset_size)
      Q['X'].observe(data[subset, :])
      Q.update('Z')
      step = (n + delay) ** (-forgetting_rate)
      Q.gradient_step('P', 'R', scale=step)
    k = P.get_parameters()
    k = np.array(k)
    means = np.zeros((20,784))
    t = R.get_parameters()
    t = np.array(t)
    t_sum = np.sum(t, axis=1)
    wts = np.zeros(20)
    ##### Finding the mean of true posterior Dirichlet distributions #####
    for i in range(20):
      wts[i] = t[0][i]/t_sum[0]
    ##### Finding the mean of true posterior Beta distributions of all mixture components #####
    # mean of beta is a0 / (a0 + a1)
    for i in range(20):
      for j in range(784):
        means[i][j] = k[0][j][i][0]/(k[0][j][i][0] + k[0][j][i][1])
    # Plotting the cluster means and weights
    plot_data(wts, means)
if __name__ == "__main__":
    main()

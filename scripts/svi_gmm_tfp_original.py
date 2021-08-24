# SVI for a GMM
# https://github.com/brendanhasz/svi-gaussian-mixture-model/blob/master/BayesianGaussianMixtureModel.ipynb
# MIT License

#pip install tf-nightly
#pip install --upgrade tfp-nightly -q

# Imports
import superimport

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from time import time

# Plot settings
#%config InlineBackend.figure_format = 'svg'

# Random seed
np.random.seed(12345)
tf.random.set_seed(12345)

# Generate some data
N = 3000
X = np.random.randn(N, 2).astype('float32')
X[:1000, :] += [2, 0]
X[1000:2000, :] -= [2, 4]
X[2000:, :] += [-2, 4]

# Plot the data
plt.plot(X[:, 0], X[:, 1], '.')
plt.axis('equal')
plt.show()

# Make a TensorFlow Dataset from that data
batch_size = 500
dataset = tf.data.Dataset.from_tensor_slices(
    (X)).shuffle(10000).batch(batch_size)


class GaussianMixtureModel(tf.keras.Model):
    """A Bayesian Gaussian mixture model.
    
    Assumes Gaussians' variances in each dimension are independent.
    
    Parameters
    ----------
    Nc : int > 0
        Number of mixture components.
    Nd : int > 0
        Number of dimensions.
    """
      
    def __init__(self, Nc, Nd):
        
        # Initialize
        super(GaussianMixtureModel, self).__init__()
        self.Nc = Nc
        self.Nd = Nd
        
        # Variational distribution variables for means
        self.locs = tf.Variable(tf.random.normal((Nc, Nd)))
        self.scales = tf.Variable(tf.pow(tf.random.gamma((Nc, Nd), 5, 5), -0.5))
        
        # Variational distribution variables for standard deviations
        self.alpha = tf.Variable(tf.random.uniform((Nc, Nd), 4., 6.))
        self.beta = tf.Variable(tf.random.uniform((Nc, Nd), 4., 6.))
        
        # Variational distribution variables for component weights
        self.counts = tf.Variable(2*tf.ones((Nc,)))

        # Prior distributions for the means
        self.mu_prior = tfd.Normal(tf.zeros((Nc, Nd)), tf.ones((Nc, Nd)))

        # Prior distributions for the standard deviations
        self.sigma_prior = tfd.Gamma(5*tf.ones((Nc, Nd)), 5*tf.ones((Nc, Nd)))
        
        # Prior distributions for the component weights
        self.theta_prior = tfd.Dirichlet(2*tf.ones((Nc,)))
        
        
        
    def call(self, x, sampling=True, independent=True):
        """Compute losses given a batch of data.
        
        Parameters
        ----------
        x : tf.Tensor
            A batch of data
        sampling : bool
            Whether to sample from the variational posterior
            distributions (if True, the default), or just use the
            mean of the variational distributions (if False).
            
        Returns
        -------
        log_likelihoods : tf.Tensor
            Log likelihood for each sample
        kl_sum : tf.Tensor
            Sum of the KL divergences between the variational
            distributions and their priors
        """
        
        # The variational distributions
        mu = tfd.Normal(self.locs, self.scales)
        sigma = tfd.Gamma(self.alpha, self.beta)
        theta = tfd.Dirichlet(self.counts)
        
        # Sample from the variational distributions
        if sampling:
            Nb = x.shape[0] #number of samples in the batch
            mu_sample = mu.sample(Nb)
            sigma_sample = tf.pow(sigma.sample(Nb), -0.5)
            theta_sample = theta.sample(Nb)
        else:
            mu_sample = tf.reshape(mu.mean(), (1, self.Nc, self.Nd))
            sigma_sample = tf.pow(tf.reshape(sigma.mean(), (1, self.Nc, self.Nd)), -0.5)
            theta_sample = tf.reshape(theta.mean(), (1, self.Nc))
        
        # The mixture density
        density = tfd.Mixture(
            cat=tfd.Categorical(probs=theta_sample),
            components=[
                tfd.MultivariateNormalDiag(loc=mu_sample[:, i, :],
                                           scale_diag=sigma_sample[:, i, :])
                for i in range(self.Nc)])
                
        # Compute the mean log likelihood
        log_likelihoods = density.log_prob(x)
        
        # Compute the KL divergence sum
        mu_div    = tf.reduce_sum(tfd.kl_divergence(mu,    self.mu_prior))
        sigma_div = tf.reduce_sum(tfd.kl_divergence(sigma, self.sigma_prior))
        theta_div = tf.reduce_sum(tfd.kl_divergence(theta, self.theta_prior))
        kl_sum = mu_div + sigma_div + theta_div
        
        # Return both losses
        return log_likelihoods, kl_sum

# A GMM with 3 components in 2 dimensions
model = GaussianMixtureModel(3, 2)

# Use the Adam optimizer
optimizer = tf.keras.optimizers.Adam(lr=1e-3)

@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        log_likelihoods, kl_sum = model(data)
        elbo_loss = kl_sum/N - tf.reduce_mean(log_likelihoods)
    gradients = tape.gradient(elbo_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
# Fit the model
EPOCHS = 1000
time_start = time()
for epoch in range(EPOCHS):
    for data in dataset:
        train_step(data)
elapsed_time = (time() - time_start)
#print('method {}'.format(method))
print(elapsed_time)
       
# Compute log likelihood at each point on a grid
Np = 100 #number of grid points
Xp, Yp = np.meshgrid(np.linspace(-6, 6, Np), np.linspace(-6, 6, Np))
Pp = np.column_stack([Xp.flatten(), Yp.flatten()]).astype('float32')
Z, _ = model(Pp, sampling=False)
Z = np.reshape(Z, (Np, Np))
        
# Show the fit mixture density
plt.figure()
plt.imshow(np.exp(Z),
           extent=(-6, 6, -6, 6),
           origin='lower')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Likelihood')

model.locs

model.trainable_variables

# Sample from the std deviation variational posterior
stds = tf.pow(tfd.Gamma(model.alpha, model.beta).sample(10000), -0.5)

# Plot the samples
plt.figure()
sns.distplot(stds[:, 0, 0])

# Sample from the mean variational posterior
means = tfd.Normal(model.locs, model.scales).sample(10000)

# Plot the mean samples for a single 
plt.figure()
sns.kdeplot(means[:, 0, 0].numpy(),
            means[:, 0, 1].numpy(),
            n_levels=10)


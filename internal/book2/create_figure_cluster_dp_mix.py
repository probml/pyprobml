import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from jax.numpy.linalg import slogdet, solve
from jax import random
from collections import namedtuple

                
@jax.jit
def log_p_of_multi_t(x, nu, mu, Sigma):
    """
    Computing the logarithm of probability density of the multivariate T distribution,
    https://en.wikipedia.org/wiki/Multivariate_t-distribution
    ---------------------------------------------------------
    x: array(dim)
        Data point that we want to evaluate log pdf at
    nu: int
        Degree of freedom of the multivariate T distribution
    mu: array(dim)
        Location parameter of the multivariate T distribution
    Sigma: array(dim, dim) 
        Positive-definite real scale matrix of the multivariate T distribution
    --------------------------------------------------------------------------
    * float
        Log probability of the multivariate T distribution at x
    """
    dim = mu.shape[0]
    # Logarithm of the normalizing constant
    l0 = gammaln((nu+dim)/2.0) - (gammaln(nu/2.0) + dim/2.0*(jnp.log(nu)+jnp.log(np.pi)) + slogdet(Sigma)[1])
    # Logarithm of the unnormalized pdf
    l1 = -(nu+dim)/2.0 * jnp.log(1 + 1/nu*(x-mu).dot(solve(Sigma, x-mu)))
    return l0 + l1


def log_predic_t(x, obs, hyper_params):
    """
    Evaluating the logarithm of probability of the posterior predictive multivariate T distribution.
    The likelihood of the observation given the parameter is Gaussian distribution.
    The prior distribution is Normal Inverse Wishart (NIW) with parameters given by hyper_params.
    ---------------------------------------------------------------------------------------------
    x: array(dim)
        Data point that we want to evalute the log probability 
    obs: array(n, dim)
        Observations that the posterior distritbuion is conditioned on
    hyper_params: () 
        The set of hyper parameters of the NIW prior
    ------------------------------------------------
    * float
        Log probability of the multivariate T distribution at x
    """
    mu0, kappa0, nu0, Sigma0 = hyper_params
    n, dim = obs.shape
    # Use the prior marginal distribution if no observation
    if n==0:
        nu_t = nu0 - dim + 1 
        mu_t = mu0
        Sigma_t = Sigma0*(kappa0+1)/(kappa0*nu_t)
        return log_p_of_multi_t(x, nu_t, mu_t, Sigma_t)
    # Update the distribution using sufficient statistics
    obs_mean = jnp.mean(obs, axis=0)
    S = (obs-obs_mean).T @ (obs-obs_mean)
    nu_n = nu0 + n
    kappa_n = kappa0 + n
    mu_n = kappa0/kappa_n*mu0 + n/kappa_n*obs_mean
    Lambda_n = Sigma0 + S + kappa0*n/kappa_n*jnp.outer(obs_mean-mu0, obs_mean-mu0)
    nu_t = nu_n - dim + 1
    mu_t = mu_n
    Sigma_t = Lambda_n*(kappa_n+1)/(kappa_n*nu_t)
    return log_p_of_multi_t(x, nu_t, mu_t, Sigma_t)


def dp_cluster(T, X, alpha, hyper_params, key):
    """
    Implementation of algorithm3 of R.M.Neal(2000)
    https://www.tandfonline.com/doi/abs/10.1080/10618600.2000.10474879
    The clustering analysis using Gaussian Dirichlet process (DP) mixture model
    ---------------------------------------------------------------------------
    T: int 
        Number of iterations of the MCMC sampling
    X: array(size_of_data, dimension)
        The array of observations
    alpha: float
        Concentration parameter of the DP
    hyper_params: object of NormalInverseWishart
        Base measure of the Dirichlet process
    key: jax.random.PRNGKey
        Seed of initial random cluster
    ----------------------------------
    * array(T, size_of_data):
        Simulation of cluster assignment
    """
    n, dim = X.shape
    Zs = []
    Cluster = namedtuple('Cluster', ["label", "members"])
    # Initialize by setting all observations to cluster0
    cluster0 = Cluster(label=0, members=list(range(n)))
    # CR is set of clusters
    CR = [cluster0]
    Z = jnp.full(n, 0) 
    new_label = 1 
    for t in range(T):
        # Update the cluster assignment for every observation
        for i in range(n):
            labels = [cluster.label for cluster in CR]
            j = labels.index(Z[i])
            CR[j].members.remove(i)
            if len(CR[j].members) == 0:
                del CR[j]
            lp0 = [jnp.log(len(cluster.members))+ log_predic_t(X[i,], jnp.atleast_2d(X[cluster.members[:],]), hyper_params) for cluster in CR]
            lp1 = [jnp.log(alpha) + log_predic_t(X[i,], jnp.empty((0, dim)), hyper_params)]
            logits = jnp.array(lp0 + lp1)
            key, subkey = random.split(key)
            k = random.categorical(subkey, logits=logits)
            if k==len(logits)-1:
                new_cluster = Cluster(label=new_label, members=[i])
                new_label += 1
                CR.append(new_cluster)
                Z = Z.at[i].set(new_cluster.label)
            else:
                CR[k].members.append(i)
                Z = Z.at[i].set(CR[k].label)
        Zs.append(Z)
    return jnp.array(Zs)


if __name__=='__main__':
    
    from jax.scipy.linalg import sqrtm
    import matplotlib.pyplot as plt
    
    from NIW import NormalInverseWishart
    from dp_mix_plot import dp_mixture_simu

    
    # Sample the generative model.
    dim = 2
    params = dict(
        loc=jnp.zeros(dim),
        mean_precision=0.05,
        df=dim + 5,
        scale=jnp.eye(dim)
    )
    niw = NormalInverseWishart(**params)
    N = 300
    alpha = 2.0
    _, X0, _, _ = dp_mixture_simu(N, alpha, niw, random.PRNGKey(3))
    X1 = random.permutation(random.PRNGKey(0), X0, axis=0)
    
    # Perform the posterior inference
    hyper_params = params.values()
    T = 110
    Zs0 = dp_cluster(T, X0, alpha, hyper_params, random.PRNGKey(0))
    Zs1 = dp_cluster(T, X1, alpha, hyper_params, random.PRNGKey(0))
    
    # plot
    bb = np.arange(0, 2 * np.pi, 0.02)
    ts = [10, 50, 100]
    Xs = [X0, X1]
    Zs = [Zs0, Zs1]
    # Different rows represents different iterations in posterior sampling;
    # different column represents different shuffling of the data.
    fig, axes = plt.subplots(3, 2)
    plt.setp(axes, xticks=[], yticks=[])
    for i in range(2):
        zs = Zs[i]
        x = Xs[i]
        for j in range(3):
            axes[j,i].plot(x[:,0], x[:, 1], ".", markersize=5)
            # The clustering after ts[j] iterations
            Z = zs[ts[j],]
            for k in jnp.unique(Z):
                xk = x[Z==k,]
                mu_k = jnp.atleast_2d(jnp.mean(xk, axis=0))
                Sig_k = (xk-mu_k).T @ (xk-mu_k) / (xk.shape[0]-1)
                Sig_root = sqrtm(Sig_k)
                circ = mu_k.T.dot(np.ones((1, len(bb)))) + Sig_root.dot(np.vstack([np.sin(bb), np.cos(bb)]))
                axes[j,i].plot(circ[0, :], circ[1, :], linewidth=2, color="k")
    plt.show()
        
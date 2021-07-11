# Author: Meduri Venkata Shivaditya
# Bernoulli mixture model for mnist digits
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import bernoulli as bern
import warnings


warnings.filterwarnings("ignore")


def bernoulli_mixture_pmf(data, means, K):
    '''To compute the probability of x for each bernouli distribution
    data = N X D matrix
    means = K X D matrix
    prob (result) = N X K matrix
    '''
    N = len(data)
    D = len(data[0])
    # compute prob(x/mean)
    # prob[i, k] for ith data point, and kth cluster/mixture distribution
    prob = np.zeros((N, K))
    for k in range(K):
            b = lambda row : np.prod(bern.pmf(row, means[k]))
            prob[:, k] = np.apply_along_axis(b, 1, data)
    return prob


def compute_responsibilities(data, weights, means, K):
    '''To compute responsibilities, or posterior probability p(z/x)
    data = N X D matrix
    weights = K dimensional vector
    means = K X D matrix
    prob or responsibilities (result) = N X K matrix
    '''
    # step 1
    # calculate the p(x/means)
    prob = bernoulli_mixture_pmf(data, means, K)

    # step 2
    # calculate the numerator of the responsibilities.s
    prob = prob * weights
    # step 3
    # calcualte the denominator of the responsibilities.s
    row_sums = prob.sum(axis=1)[:, np.newaxis]

    # step 4
    # calculate the responsibilities.s
    try:
        prob = prob / row_sums
        prob = np.nan_to_num(prob)
        return prob
    except ZeroDivisionError:
        print("Division by zero occured in reponsibility calculations!")


def m_step(data, responsibilities, K):
    '''Re-estimate the parameters using the current responsibilities
    data = N X D matrix
    responsibilities = N X K matrix
    return revised weights (K vector) and means (K X D matrix)
    '''
    N = len(data)
    D = len(data[0])

    Nk = np.sum(responsibilities, axis=0)
    mus = np.empty((K, D))

    for k in range(K):
        mus[k] = np.sum(responsibilities[:, k][:, np.newaxis] * data, axis=0)  # sum is over N data points
        try:
            mus[k] = mus[k] / Nk[k]
        except ZeroDivisionError:
            print("Division by zero occured in Mixture of Bernoulli Dist M-Step!")
            break

    return (Nk / N, mus)


def expected_log_likelihood(data, weights, means, K):
    '''To compute expectation of the loglikelihood of Mixture of Beroullie distributions
    Since computing E(LL) requires computing responsibilities, this function does a double-duty
    to return responsibilities too
    '''
    N = len(data)
    responsibilities = compute_responsibilities(data, weights, means, K)
    ll = 0
    sumK = np.zeros(N)
    for k in range(K):
        b = lambda row : np.log(bern.pmf(row, np.absolute(means[k])))
        temp1 = np.apply_along_axis(b, 1, data)
        sumK += responsibilities[:, k] * (np.log(np.absolute(weights[k])) + np.sum(temp1, axis=1))
        sumK = np.nan_to_num(sumK)
    ll += np.sum(sumK)
    return (ll, responsibilities)


def mixOfBernoulliEM(data, K, maxiters=1000, relgap=1e-4, verbose=False):
    '''EM algo fo Mixture of Bernoulli Distributions'''

    N = len(data)
    D = len(data[0])

    # initalize
    #initializing weigths randomly
    init_weights = np.random.uniform(100, 200, K)
    tot = np.sum(init_weights)
    init_weights = init_weights / tot
    init_means = np.full((K, D), 1.0 / K)
    weights = init_weights[:]
    means = init_means[:]
    ll, responsibilities = expected_log_likelihood(data, weights, means, K)
    ll_old = ll
    for i in range(maxiters):
        print("iteration {}:".format(i))
        print("   {}:".format(weights))
        print("   {:.6}".format(ll))

        # E Step: calculate responsibilities
        # Skip, rolled into log likelihood calc
        # For 0th step, done as part of initialization

        # M Step
        weights, means = m_step(data, responsibilities, K)

        # convergence check
        ll, responsibilities = expected_log_likelihood(data, weights, means, K)
        if np.abs(ll - ll_old) < relgap:
            print("Relative gap:{:.8} at iternations {}".format(ll - ll_old, i))
            break
        else:
            ll_old = ll
    return (weights, means)

def mnist_data(n):
    #Downloading data from tensorflow datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = (x_train>0).astype('int') #Coverting to binary
    x_train = x[np.random.randint(x.shape[0], size=n)]
    x_train = x_train.reshape((n, 784))
    return x_train

def plot_data(wts, mns):
    fig, ax = plt.subplots(4, 5)
    k = 0
    for i in range(4):
        for j in range(5):
            ax[i][j].imshow(mns[k].reshape(28, 28), cmap=plt.cm.gray)
            ax[i][j].set_title("%1.2f" % wts[k])
            ax[i][j].axis("off")
            k = k + 1
    fig.tight_layout(pad=1.0)
    plt.savefig("mixBernoulliMnist.png", dpi=300)
    plt.show()

def main():
    np.random.seed(0)
    data = mnist_data(n=1000)# subsamplethe MNIST dataset
    K = 20
    wts, mns = mixOfBernoulliEM(data, K, maxiters=5, relgap=1e-15, verbose=True)
    plot_data(wts, mns)
    
if __name__ == "__main__":
    main() 

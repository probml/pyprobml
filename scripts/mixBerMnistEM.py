# Author: Meduri Venkata Shivaditya
# Bernoulli mixture model for mnist digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def bernoulli(data, means):
    '''To compute the probability of x for each bernouli distribution
    data = N X D matrix
    means = K X D matrix
    prob (result) = N X K matrix
    '''
    N = len(data)
    K = len(means)
    # compute prob(x/mean)
    # prob[i, k] for ith data point, and kth cluster/mixture distribution
    prob = np.zeros((N, K))

    for i in range(N):
        for k in range(K):
            prob[i, k] = np.prod((means[k] ** data[i]) * ((1 - means[k]) ** (1 - data[i])))

    return prob


def respBernoulli(data, weights, means):
    '''To compute responsibilities, or posterior probability p(z/x)
    data = N X D matrix
    weights = K dimensional vector
    means = K X D matrix
    prob or resp (result) = N X K matrix
    '''
    # step 1
    # calculate the p(x/means)
    prob = bernoulli(data, means)

    # step 2
    # calculate the numerator of the resp.s
    prob = prob * weights

    # step 3
    # calcualte the denominator of the resp.s
    row_sums = prob.sum(axis=1)[:, np.newaxis]

    # step 4
    # calculate the resp.s
    try:
        prob = prob / row_sums
        return prob
    except ZeroDivisionError:
        print("Division by zero occured in reponsibility calculations!")


def bernoulliMStep(data, resp):
    '''Re-estimate the parameters using the current responsibilities
    data = N X D matrix
    resp = N X K matrix
    return revised weights (K vector) and means (K X D matrix)
    '''
    N = len(data)
    D = len(data[0])
    K = len(resp[0])

    Nk = np.sum(resp, axis=0)
    mus = np.empty((K, D))

    for k in range(K):
        mus[k] = np.sum(resp[:, k][:, np.newaxis] * data, axis=0)  # sum is over N data points
        try:
            mus[k] = mus[k] / Nk[k]
        except ZeroDivisionError:
            print("Division by zero occured in Mixture of Bernoulli Dist M-Step!")
            break

    return (Nk / N, mus)


def llBernoulli(data, weights, means):
    '''To compute expectation of the loglikelihood of Mixture of Beroullie distributions
    Since computing E(LL) requires computing responsibilities, this function does a double-duty
    to return responsibilities too
    '''
    N = len(data)
    K = len(means)

    resp = respBernoulli(data, weights, means)

    ll = 0
    for i in range(N):
        sumK = 0
        for k in range(K):
            try:
                temp1 = ((means[k] ** data[i]) * ((1 - means[k]) ** (1 - data[i])))
                temp1 = np.log(temp1.clip(min=1e-50))

            except:
                print("Problem computing log(probability)")
            sumK += resp[i, k] * (np.log(weights[k]) + np.sum(temp1))
        ll += sumK

    return (ll, resp)


def mixOfBernoulliEM(data, init_weights, init_means, maxiters=1000, relgap=1e-4, verbose=False):
    '''EM algo fo Mixture of Bernoulli Distributions'''
    N = len(data)
    D = len(data[0])
    K = len(init_means)

    # initalize
    weights = init_weights[:]
    means = init_means[:]
    ll, resp = llBernoulli(data, weights, means)
    ll_old = ll

    for i in range(maxiters):
        if verbose and (i % 5 == 0):
            print("iteration {}:".format(i))
            print("   {}:".format(weights))
            print("   {:.6}".format(ll))

        # E Step: calculate resps
        # Skip, rolled into log likelihood calc
        # For 0th step, done as part of initialization

        # M Step
        weights, means = bernoulliMStep(data, resp)

        # convergence check
        ll, resp = llBernoulli(data, weights, means)
        if np.abs(ll - ll_old) < relgap:
            print("Relative gap:{:.8} at iternations {}".format(ll - ll_old, i))
            break
        else:
            ll_old = ll

    return (weights, means)
def main():
    #Importing the MNIST dataset
    np.random.seed(0)
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    x = (mnist.data / 128).astype('int') #Coverting to binary
    y = mnist.target
    x = np.array(x)
    y = np.array(y)
    y = y.reshape((70000, 1))
    np.random.seed(0)
    x_train = x[np.random.randint(x.shape[0], size=1000)]
    K = 20
    D = 784
    #initializing weigths randomly
    initWts = np.random.uniform(1, 20, K)
    tot = np.sum(initWts)
    initWts = initWts / tot
    initMeans = np.full((K, D), 1.0 / K)
    wts, mns = mixOfBernoulliEM(x_train[:1000], initWts, initMeans, maxiters=10, relgap=1e-15, verbose=True)
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
if __name__ == "__main__":
    main()

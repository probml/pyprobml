import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os

def logdet(Sigma):
    return np.log2(Sigma)


def gaussProb(X, mu, Sigma):
    d = 1
    X = X.reshape(X.shape[0], d)
    X = X - np.transpose(mu)
    logp = -0.5*np.sum(np.multiply((X/(Sigma)), X), 1)
    logZ = (d/2)*np.log(2*np.pi) + 0.5*logdet(Sigma)
    logp = logp - logZ
    p = np.exp(logp)
    return p


def postDensityIntervals():
    def f(x): return gaussProb(x, 0, 1) + gaussProb(x, 6, 1)
    domain = np.arange(-4, 10.001, 0.001)
    plt.plot(domain, f(domain), '-r')
    plt.fill_between(domain, f(domain), color='blue')
    plt.fill_between(np.arange(-4, -1.999, 0.001),
                     f(np.arange(-4, -1.999, 0.001)), color='white')
    plt.fill_between(np.arange(8, 10.001, 0.001), f(
        np.arange(8, 10.001, 0.001)), color='white')
    plt.annotate(r'$\alpha /2$', xytext=(-3.5, 0.11), xy=(-2.3, 0.015),
                 arrowprops=dict(facecolor='black'))
    plt.annotate(r'$\alpha /2$', xytext=(9.5, 0.11), xy=(8.3, 0.015),
                 arrowprops=dict(facecolor='black'))
    plt.ylim(0, 0.5)
    plt.savefig('../figures/centralInterval')
    plt.show()

    plt.plot(domain, f(domain), '-r')
    plt.fill_between(domain, f(domain), color='blue')
    plt.fill_between(np.arange(-4, -1.43992, 0.001),
                     f(np.arange(-4, -1.43992, 0.001)), color='white')
    plt.fill_between(np.arange(7.37782, 10.001, 0.001), f(
        np.arange(7.37782, 10.001, 0.001)), color='white')
    plt.plot(domain, [0.15 for i in range(0, 14001)], 'b-')
    plt.fill_between(np.arange(1.3544, 4.5837, 0.001), f(
        np.arange(1.3544, 4.5837, 0.001)), color='white')
    plt.yticks([0.15], ["pMIN"])
    plt.ylim(0, 0.5)
    plt.savefig('../figures/HPD')

    plt.show()

def main():
    postDensityIntervals()


if __name__ == '__main__':
    if os.path.isdir('scripts'):
        os.chdir('scripts')
    main()

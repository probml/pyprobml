import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import os


def demo(priorVar):
    np.random.seed(1)
    n = 150
    m = 10
    Nobs = m
    D = n + 1
    Nhid = D - Nobs
    xs = np.linspace(0, 1, D).reshape(1, 151)
    perm = np.random.permutation(D).reshape(151, 1)
    obsNdx = perm[: 10]
    hidNdx = np.setdiff1d(np.arange(0, 151, 1), obsNdx)

    xobs = np.random.randn(Nobs, 1)
    obsNoiseVar = 1
    y = xobs + np.sqrt(obsNoiseVar)*np.random.randn(Nobs, 1)
    # checkArr = np.tile([-1, 2, -1], (n-1, 1))
    # print(len(np.array([0, 1, 2])))
    L = (0.5*scipy.sparse.diags([-1, 2, -1], [0, 1, 2], (n-1, n+1))).toarray()
    Lambda = 1 / priorVar
    L = L * Lambda
    L1 = L[:, hidNdx]
    L2 = L[:, obsNdx].reshape(149, 10)

    B11 = np.dot(np.transpose(L1), L1)
    B12 = np.dot(np.transpose(L1), L2)
    B21 = np.transpose(B12)
    postDist_mu = -np.dot(np.dot(np.linalg.inv(B11), B12), xobs)
    postDist_Sigma = np.linalg.inv(B11)

    mu = np.zeros((D, 1))
    mu[hidNdx] = -np.dot(np.dot(np.linalg.inv(B11), B12), xobs)
    mu[obsNdx.reshape(10,)] = xobs.reshape(10, 1)
    Sigma = 1e-5*np.eye(D, D)
    inverseB11 = np.linalg.inv(B11)
    for i in range(0, 141):
        for j in range(0, 141):
            Sigma[i, j] = inverseB11[i, j]
    postDist_mu = mu
    postDist_Sigma = Sigma
    Str = 'obsVar=0, priorVar='+str(round(priorVar, 3))
    makePlots(postDist_mu, postDist_Sigma, xs, xobs, xobs, hidNdx, obsNdx, Str)
    fname = 'gaussInterpNoisyDemoStable_obsVar' + \
        str(round(100*0))+'_priorVar'+str(round(100*priorVar))
    plt.savefig(r'../figures/'+fname)
    plt.show()

    C = obsNoiseVar * np.eye(Nobs, Nobs)
    row1 = np.concatenate((B11, B12), axis=1)
    row2 = np.concatenate((B21, (np.dot(np.dot(B21, np.linalg.inv(B11)),  B12) + np.linalg.inv(C))), axis=1)
    final = np.concatenate((row1, row2), axis=0)
    # (np.dot(np.dot(B21, np.linalg.inv(B11)),  B12) + np.linalg.inv(C))
    GammaInv = final
    Gamma = np.linalg.inv(GammaInv)
    postDist_Sigma = Gamma
    x  = np.concatenate((np.zeros((D-Nobs,1)), y))
    #print(Gamma.shape)
    postDist_mu = np.dot(Gamma, x)



def makePlots(postDist_mu, postDist_Sigma, xs, xobs, y, hidNdx, obsNdx, str):
    D = len(hidNdx) + len(obsNdx)
    mu = postDist_mu.reshape(151, )
    S2 = np.diag(postDist_Sigma)
    # plt.fill(np.array(np.transpose(xs), np.flip(np.transpose(xs)))
    xs = xs.reshape(151, 1)
    plt.plot(xs[obsNdx].reshape(10, 1), y, 'bx')
    plt.plot(xs, mu, 'r-')
    plt.title(str)
    for i in range(1, 3):
        fs = np.random.multivariate_normal(mu, postDist_Sigma)
        plt.plot(xs, fs, 'k-')


def main():
    if os.path.isdir('scripts'):
        os.chdir('scripts')
    demo(0.1)


if __name__ == '__main__':
    main()

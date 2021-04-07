# Author: Meduri Venkata Shivaditya
# Illustration of data imputation using an MVN.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix

def is_pos_def(x):
    #Check if the matrix is positive definite
    j = np.linalg.eigvals(x)
    return np.all(j>0)

def gaussSample(mu, sigma, n):
    # Returns n samples (in the rows) from a multivariate Gaussian distribution

    a = np.linalg.cholesky(sigma)
    z = np.random.randn(len(mu), n)
    k = np.dot(a, z)
    return np.transpose(mu + k)

def gaussCondition(m, s, v, vv):
    # p(xh | xv = visValues)

    d = len(m)
    j = np.array(range(d))
    h = np.setdiff1d(j, v)
    if len(h)==0:
        mugivh = np.array([])
        sigivh = np.array([])
    elif len(v) == 0:
        mugivh = m
        sigivh = s
    else:
        shh = np.zeros((len(h), len(h)))
        for lr, uu in enumerate(h):
            for lc, jj in enumerate(h):
                shh[lr, lc] = s[uu, jj]
        shv = np.zeros((len(h), len(v)))
        for lr, uu in enumerate(h):
            for lc, jj in enumerate(v):
                shv[lr, lc] = s[uu, jj]
        svv = np.zeros((len(v), len(v)))
        for lr, uu in enumerate(v):
            for lc, jj in enumerate(v):
                svv[lr, lc] = s[uu, jj]
        svvin = np.linalg.inv(svv)
        vvl = len(vv)
        mugivh = m[h] + np.dot(shv, np.dot(svvin, (vv.reshape((vvl,1))-m[v].reshape((vvl, 1)))))
        sigivh = shh - np.dot(shv, np.dot(svvin, np.transpose(shv)))
    return mugivh, sigivh

def gaussImpute(mu, sigma, x):
    #Perform Gauss Imputation to the matrix x using mu and sigma
    #Fill in NaN entries of X using posterior mode on each row
    #Xc(i,j) = E[X(i,j) | D]

    n, d = x.shape
    xc = np.copy(x)
    v = np.zeros((n, d))
    for i in range(n):
        hn = np.argwhere(np.isnan(x[i, :]))
        vn = np.argwhere(~np.isnan(x[i, :]))
        vv = np.zeros(len(vn))
        for tc, h in enumerate(vn):
            vv[tc] = x[i, h]
        m_hgv, s_hgv = gaussCondition(mu, sigma, vn, vv)
        for rr, h in enumerate(hn):
            xc[i, h] = m_hgv[rr]
    #    for rr, h in enumerate(hn):
    #        v[i, h] = s_hgv[rr]
    return xc


def hinton(matrix, max_weight=None, ax=None, pl = None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else pl.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('white')
    ax.set_aspect('equal', 'box')

    for (x, y), w in np.ndenumerate(matrix):
        color = 'lawngreen' if w > 0 else 'royalblue'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    ax.grid(linestyle='--')
    ax.autoscale_view()
    ax.invert_yaxis()

def main():
    np.random.seed(12)
    d = 8
    n = 10
    pcMissing = 0.5
    mu = np.random.randn(d, 1)
    sigma = make_spd_matrix(n_dim=d)  # Generate a random positive semi-definite matrix
    # test if the matrix is positive definite
    # print(is_pos_def(sigma))
    Xfull = gaussSample(mu, sigma, n)
    missing = np.random.rand(n, d) < pcMissing
    Xmiss = np.copy(Xfull)
    Xmiss[missing] = np.nan
    xc = gaussImpute(mu, sigma, Xmiss)
    xmiss0 = np.copy(Xmiss)
    for g in np.argwhere(np.isnan(Xmiss)):
        xmiss0[g[0], g[1]] = 0
    p1 = plt.figure(1)
    hinton(xmiss0, pl=p1)
    p1.suptitle('Observed')
    p1.savefig("Hinton_Observed.png", dpi=300)
    p2 = plt.figure(2)
    hinton(Xfull, pl=p2)
    p2.suptitle('Truth')
    p2.savefig("Hinton_Truth.png", dpi=300)
    p3 = plt.figure(3)
    hinton(xc, pl=p3)
    p3.suptitle('imputation with true params')
    p3.savefig("Hinton_ImputationWithTrueParams.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

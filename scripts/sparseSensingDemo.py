import numpy as np
import math
import sys
from scipy.linalg import orth
from numpy.linalg import norm
from scipy.sparse.linalg import cg, LinearOperator


def AXfunc(A, At, d1, d2, p1, p2, p3):
    def matvec(vec):
        n = vec.shape[0] // 2
        x1 = vec[:n]
        x2 = vec[n:]

        return np.hstack([At.dot(A.dot(x1) * 2) + d1 * x1 + d2 * x2,
                          d2 * x1 + d1 * x2])
    N = 2 * d1.shape[0]
    return LinearOperator((N, N), matvec=matvec)


def MXfunc(A, At, d1, d2, p1, p2, p3):
    def matvec(vec):
        n = vec.shape[0] // 2
        x1 = vec[:n]
        x2 = vec[n:]

        return np.hstack([p1 * x1 - p2 * x2,
                          -p2 * x1 + p3 * x2])

    N = 2 * p1.shape[0]
    return LinearOperator((N, N), matvec=matvec)


def l1ls(A, y, lmbda, x0=None, At=None, m=None, n=None, tar_gap=1e-3, quiet=False, eta=1e-3, pcgmaxi=5000):

    # l1ls
    # Credits: https://github.com/Networks-Learning
    # Author: https://github.com/musically-ut

    At = A.transpose() if At is None else At
    m = A.shape[0] if m is None else m
    n = A.shape[1] if n is None else n
    MU = 2
    MAX_NT_ITER = 400
    ALPHA = 0.01
    BETA = 0.5
    MAX_LS_ITER = 100
    t0 = min(max(1, 1/lmbda), 2 * n / 1e-3)
    x = np.zeros(n) if x0 is None else x0.ravel()
    y = y.ravel()
    status, history = 'Failed', []
    u = np.ones(n)
    t = t0
    reltol = tar_gap
    f = np.hstack((x - u, - x - u))
    pobjs, dobjs, sts, pflgs = [], [], [], []
    pobj, dobj, s, pflg = np.inf, -np.inf, np.inf, 0
    ntiter, lsiter = 0, 0
    normg = 0
    dxu = np.zeros(2*n)
    diagxtx = 2 * np.ones(n)

    for ntiter in range(0, MAX_NT_ITER):
        z = A.dot(x) - y
        nu = 2 * z
        maxAnu = norm(At.dot(nu), np.inf)
        if maxAnu > lmbda:
            nu = nu * lmbda / maxAnu
        pobj = z.dot(z) + lmbda*norm(x, 1)
        dobj = max(-0.25 * nu.dot(nu) - nu.dot(y), dobj)
        gap = pobj - dobj
        pobjs.append(pobj)
        dobjs.append(dobj)
        sts.append(s)
        pflgs.append(pflg)
        if (gap / dobj) < reltol:
            status = 'Solved'
            history = np.vstack([np.asarray(pobjs) - np.asarray(dobjs),
                                 pobjs, dobjs, sts, pflgs]).transpose()

            break
        if s >= 0.5:
            t = max(min(2 * n * MU / gap, MU * t), t)

        # Calculate Newton step
        q1, q2 = 1 / (u + x), 1 / (u - x)
        d1, d2 = (q1 ** 2 + q2 ** 2) / t, (q1 ** 2 - q2 ** 2) / t

        # calculate the gradient
        gradphi = np.hstack([At.dot(2 * z) - (q1 - q2) / t,
                             lmbda * np.ones(n) - (q1 + q2) / t])

        # calculate vectors to be used in the preconditioner
        prb = diagxtx + d1
        prs = prb.dot(d1) - (d2 ** 2)

        # set pcg tolerange (relative)
        normg = norm(gradphi)
        pcgtol = min(1e-1, eta * gap / min(1, normg))

        p1, p2, p3 = d1 / prs, d2 / prs, prb / prs
        dxu_old = dxu

        [dxu, info] = cg(AXfunc(A, At, d1, d2, p1, p2, p3),
                         -gradphi, x0=dxu, tol=pcgtol, maxiter=pcgmaxi,
                         M=MXfunc(A, At, d1, d2, p1, p2, p3))

        # This is to increase the tolerance of the underlying PCG if
        # it converges to the same solution without offering an increase
        # in the solution of the actual problem
        if info == 0 and np.all(dxu_old == dxu):
            pcgtol *= 0.1
            pflg = 0
        elif info < 0:
            pflg = -1
            raise TypeError('Incorrectly formulated problem.'
                            'Could not run PCG on it.')
        elif info > 0:
            pflg = 1
            if not quiet:
                print('Could not converge PCG after {} iterations.'
                      ''.format(info))
        else:
            pflg = 0

        dx, du = dxu[:n], dxu[n:]

        # Backtracking line search
        phi = z.dot(z) + lmbda * np.sum(u) - np.sum(np.log(-f)) / t
        s = 1.0
        gdx = gradphi.dot(dxu)
        for lsiter in range(MAX_LS_ITER):
            newx, newu = x + s * dx, u + s * du
            newf = np.hstack([newx - newu, -newx - newu])
            if np.max(newf) < 0:
                newz = A.dot(newx) - y
                newphi = newz.dot(newz) + \
                    lmbda * np.sum(newu) - np.sum(np.log(-newf)) / t
                if newphi - phi <= ALPHA * s * gdx:
                    break
            s = BETA * s
        else:
            if not quiet:
                print('MAX_LS_ITER exceeded in BLS')
            status = 'Failed'
            history = np.vstack([np.asarray(pobjs) - np.asarray(dobjs),
                                 pobjs, dobjs, sts, pflgs]).transpose()
            break

        x, u, f = newx, newu, newf
    else:
        status = 'Failed'
        history = np.vstack([np.asarray(pobjs) - np.asarray(dobjs),
                             pobjs, dobjs, sts, pflgs]).transpose()

    # Reshape x if the original array was a 2D
    if x0 is not None:
        x = x.reshape(*x0.shape)

    return (x, status, history)


np.set_printoptions(threshold=sys.maxsize)
np.random.seed(0)
n = int(math.pow(2, 12))
k = int(math.pow(2, 10))
n_spikes = 160

f = np.zeros((n, 1))
q = np.random.permutation(n)

f[q[0: n_spikes]] = np.sign(np.random.randn(n_spikes, 1))

R = np.random.randn(k, n)
R = np.transpose(orth(np.transpose(R)))


sigma = 0.01
y = np.dot(R, f) + sigma*np.random.randn(k, 1)

tau = 0.1*np.amax(np.absolute(np.dot(np.transpose(R), y)))

x_l1_ls, status, history = l1ls(R, y, 2*tau, tar_gap=0.01)

w = x_l1_ls
aw = np.absolute(w)
zz = np.nonzero(np.absolute(w) <= 0.01 * np.amax(aw))
ndx = np.setdiff1d(np.arange(0, n), zz)
wdebiased = np.zeros((n, 1))
X = R
print(y.shape)
for i in ndx:
    wdebiased[i] = np.divide(y.reshape(1024,), (X[:, 0]))  # flag

"""Gaussian process (GP) using a cubic spline kernel.

Given a GP prior on the scalar function f(t), and noisy Gaussian observations
of the form y(t) ~ N(f(t), sigmaf) and dy(t) ~ N(df(t), sigmadf), where df
is the derivative of f, the posterior p(f, df) is also a GP.
We use a cubic spline kernel for the GP.
For general info on cubic splines, see sec 6.3 of "Gaussian processes
for machine learning", Rasmussen and Williams
(http://www.gaussianprocess.org/gpml/).

This code implements the version described in this paper:
 "Probabilistic line searches for stochastic optimization",
 M. Mahsereci and P. Hennig, NIPS 2015 (http://arxiv.org/abs/1502.02846)
 
The code is a direct translation from the matlab function probLineSearch.m.
It computes the posterior mean and variance of f, as well various
derivatives.
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
    
class CubicSplineGaussianProcess(object):
    """Gaussian process based on cubic spline."""
    def __init__(self, sigmaf, sigmadf):
        # Currently we assume the noise terms are fixed.
        self.sigmaf = sigmaf
        self.sigmadf = sigmadf
        # Create storage for mean function and its derivatives.
        self.m_fun = None
        self.d1m_fun = None
        self.d2m_fun = None
        self.d3m_fun = None
        # Create storage for variance function and its derivatives.
        self.v_fun = None
        self.vd_fun = None
        self.dvd_fun = None
        # Create storage for covariance function and its derivatives.
        self.v0f_fun = None
        self.vd0f_fun = None
        self.v0df_fun = None
        self.vd0df_fun = None
        
    def compute_posterior(self, tv, yv, dyv):
        """Update posterior over function given batch of data.
        
        Args:
            tv: N-vector of input locations.
            yv: N-vector of noisy function values (at tv).
            dyv: N-vector of noisy derivative values (at tv).
        """
        # TODO: implement faster O(N) Kalman filtering method. See eg.
        # Kohn and Ansley (1987), "A new algorithm for spline smoothing
        # based on smoothing a sotchastic process".
        
        # Implement part of eqn 7 of Mahsereci'15
        gram = make_gram_matrix(tv, self.sigmaf, self.sigmadf)
        yy = np.r_[yv, dyv]  # All the observed data.
        v = linalg.solve(gram, yy)  # G^{-1} * yy, column vector
        
        # Define local functions to make math more readable.
        def k(a, b):
            return k_spline(a, b)
        def dk(a, b):
            return dk_spline(a, b)
        def ddk(a, b):
            return ddk_spline(a, b)
        def dddk(a, b):
            return dddk_spline(a, b)
        def kd(a, b):
            return kd_spline(a, b)
        def dkd(a, b):
            return dkd_spline(a, b)
        def ddkd(a, b):
            return ddkd_spline(a, b)
        def dddkd(a, b):
            return dddkd_spline(a, b)
            
        # Posterior mean of f is given by
        # mu(t) = m(t) = ip([k(tv, t); kd(tv, t)], v)
        # To find the minimum of f in each 'cell', we need to compute
        # mu'(t) = d1dm, mu''(t) = d2m, mu'''(t) = d3m. These are given by
        # d1m(t) = ip([dk(tvals, t); dkd(tvals, t)], v), etc
        self.m_fun = lambda t: np.inner(v, np.r_[k(t, tv), kd(t, tv)])
        self.d1m_fun = lambda t: np.inner(v, np.r_[dk(t, tv), dkd(t, tv)])
        self.d2m_fun = lambda t: np.inner(v, np.r_[ddk(t, tv), ddkd(t, tv)])
        self.d3m_fun = lambda t: np.inner(v, np.r_[dddk(t, tv), dddkd(t, tv)])
        
        # posterior covariance v(t,t') = cov[f(t), f(t')] is given by eqn 7:
        # v(t,t') = k(t,t') - ip([k(t,tv); kd(t, tv)],
        #                        inv(G) * [k(t',tv); kd(t',tv)])
        # We use this formula below.
        # Note that tv is a row vector, so we concenate using r_
        
        # posterior marginal variance v(t) = v(t,t)
        self.v_fun = lambda t: k(t, t) - np.inner(np.r_[k(t,tv), kd(t,tv)],
            linalg.solve(gram, np.r_[k(t,tv), kd(t,tv)]))
            
        # vd(t) = d/dt' v(t,t') 
        self.vd_fun = lambda t: kd(t, t) - np.inner(np.r_[k(t,tv), kd(t,tv)],
            linalg.solve(gram, np.r_[dk(t,tv), dkd(t,tv)]))
            
        # dvd(t) = d/dt d/dt' v(t,t')
        self.dvd_fun = lambda t: dkd(t, t) - np.inner(np.r_[dk(t,tv), dkd(t,tv)],
            linalg.solve(gram, np.r_[dk(t,tv), dkd(t,tv)]))
        
        # v0f(t) = cov[f(t), f(0)] 
        self.v0f_fun = lambda t: k(0, t) - np.inner(np.r_[k(0,tv), kd(0,tv)],
            linalg.solve(gram, np.r_[k(t,tv), kd(t,tv)]))
        
        # vd0f(t) 
        self.vd0f_fun = lambda t: dk(0, t) - np.inner(np.r_[dk(0,tv), dkd(0,tv)],
            linalg.solve(gram, np.r_[k(t,tv), kd(t,tv)]))
            
        # v0df(t) 
        self.v0df_fun = lambda t: kd(0, t) - np.inner(np.r_[k(0,tv), kd(0,tv)],
            linalg.solve(gram, np.r_[dk(t,tv), dkd(t,tv)]))
        
        # vd0df(t)
        self.vd0df_fun = lambda t: dkd(0, t) - np.inner(np.r_[dk(0,tv), dkd(0,tv)],
            linalg.solve(gram, np.r_[dk(t,tv), dkd(t,tv)]))
            
def make_gram_matrix(tv, sigmaf, sigmadf):
    # Equation 7 of Mahsereci'15
    n = len(tv)
    kTT = np.zeros([n, n])
    kdTT = np.zeros([n, n])
    dkdTT = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            kTT[i, j] = k_spline(tv[i], tv[j])
            kdTT[i, j] = kd_spline(tv[i], tv[j])
            dkdTT[i, j] = dkd_spline(tv[i], tv[j])

    sigma = np.concatenate([sigmaf**2 * np.ones(n),
                            sigmadf**2 * np.ones(n)])
    gram = np.diag(sigma) + \
        np.bmat([[kTT, kdTT], [kdTT.T, dkdTT]])
    gram = np.asarray(gram)
    return gram
    

_offset = 10  # tau in paper, needed for numerical stability

# Cubic spline kernel function and its derivatives
def k_spline(a, b):
    # Eqn 4 of Mahsereci'15, eqn 6.28 of Rasmussen'06
    v = np.minimum(a + _offset, b + _offset)
    val = ((1.0 / 3) * np.power(v, 3) + 
            (1.0 / 2) * np.abs(a-b) * np.power(v, 2))
    return val

def kd_spline(a, b):
    # Eqn 6, d/dt' k(t,t')
    aa = a + _offset  
    bb = b + _offset  
    val = (a < b) * np.power(aa, 2)/2 + \
        (a >= b) * (aa * bb - 0.5 * np.power(bb, 2))
    return val
    
def dk_spline(a, b):
    # Eqn 6, d/dt k(t,t') 
    aa = a + _offset  
    bb = b + _offset  
    val = (a > b) * np.power(bb, 2)/2 + \
        (a <= b) * (aa * bb - 0.5 * np.power(aa, 2))
    return val
    
def dkd_spline(a, b):
    # Eqn 6, d^2/dt dt' k(t,t')
    return np.minimum(a + _offset, b + _offset)
    
def ddk_spline(a, b):
    # Eqn 8, d^2/dt^2 k(t,t')
    return (a <= b) * (b-a)

def ddkd_spline(a, b):
    # Eqn 8, d^2/dt^2 d/dt' k(t,t')
    return (a <= b) 
    
def dddk_spline(a, b):
    # Eqn 8, d^3/dt^3 k(t,t')
    return -(a <= b) 
    
def dddkd_spline(a, b):
    # Eqn 8, d^3/dt^3 d/dt' k(t,t')
    return np.zeros(max(len(a), len(b)))
    
#####

def demo():
    """Create a 1d function and iteratively try to infer it."""
    #ts = np.array([0, 1, 3, 4.5]) # 
    #g = make_gram_matrix(ts, 1, 1)
    
    fun = lambda t: 2*np.sin(t) + 0.5
    deriv = lambda t: 2*np.cos(t)
    tgrid = np.linspace(0, 5, 20)
    fvals = [fun(t) for t in tgrid]
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.plot(tgrid, fvals)
    ax.set_title('true function')
    
    ts = np.array([0, 1, 3, 4.5]) # steps along search direction
    # for reproducability with matlab, we fix the random noise
    ynoise = np.array([ 0.3273,    0.1746,   -0.1867,    0.7258])
    dynoise = np.array([-0.5883,    2.1832,   -0.1364,    0.1139])
    
    t_hist = []
    y_hist = []
    dy_hist = []
    sigmaf = 1
    sigmadf = 1
    fig = plt.figure()
    plotnum = 1
    gp = CubicSplineGaussianProcess(sigmaf, sigmadf)
    for i in range(len(ts)):
        t = ts[i]
        y = fun(t) + ynoise[i]
        dy = deriv(t) + dynoise[i]
        
        t_hist.append(t)
        y_hist.append(y)
        dy_hist.append(dy)
        
        gp.compute_posterior(np.array(t_hist), np.array(y_hist),
                            np.array(dy_hist))
        
        mgrid = [gp.m_fun(t) for t in tgrid]
        sgrid = [np.sqrt(gp.v_fun(t)) for t in tgrid]
        ax = fig.add_subplot(2,2,plotnum)
        plotnum += 1
        ax.errorbar(tgrid, mgrid, yerr=sgrid)
        ax.errorbar(t_hist, y_hist, yerr=sigmaf*np.ones(len(t_hist)), fmt='o')
        ax.set_title('estimate at iteration {}'.format(i))
        
        print 'mgrid at iter {}, {}'.format(i, mgrid)
        print 'sgrid at iter {}, {}'.format(i, sgrid)
        
            
def main():
    mgrid = demo()
    plt.show()
    
if __name__ == "__main__":
    main()



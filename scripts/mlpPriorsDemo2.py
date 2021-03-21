import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=10)

class Prior():
    def __init__(self, alpha, indx=None):
        self.alpha = alpha
        self.indx = indx
        
class Net():
    def __init__(self, net_type, nin, nhidden, nout, nwts, outfunc, 
                 alpha=None, indx=None, w1=None, b1=None, w2=None, b2=None, beta=None):
        self.net_type = net_type
        self.nin = nin
        self.nhidden = nhidden
        self.nout = nout
        self.nwts = nwts
        self.alpha = alpha
        self.indx = indx
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.beta = beta
        
        outfns = ['linear', 'logistic', 'softmax']
        if outfunc in outfns:
            self.outfunc = outfunc
        else:
            raise ValueError('Undefined output function. Exiting.')
        
    
def MLPprior(nin, nhidden, nout, aw1, ab1, aw2, ab2):
    nextra = nhidden + (nhidden+1)*nout
    nwts = nin*nhidden + nextra
    
    if np.isscalar(aw1):
        indx = np.hstack((np.ones((1, nin*nhidden)), np.zeros((1, nextra)))).T
    elif aw1.shape == (1, nin):
        indx = np.kron(np.ones((nhidden, 1)), np.identity(nin))
        indx = np.hstack((indx, np.zeros((nextra, nin))))
    else:
        print('Parameter aw1 of invalid dimensions')
    
    extra = np.zeros((nwts, 3))
    mark1 = nin*nhidden
    mark2 = mark1 + nhidden
    extra[mark1:mark2, 0] = np.ones((nhidden, 1)).T
    mark3 = mark2 + nhidden*nout
    extra[mark2:mark3, 1] = np.ones((nhidden*nout, 1)).T
    mark4 = mark3 + nout
    extra[mark3:mark4, 2] = np.ones((nout, 1)).T
    indx = np.hstack((indx, extra))
    alpha = np.hstack((aw1, ab1, aw2, ab2)).T
    prior = Prior(alpha, indx)
    return prior

def MLP(nin, nhidden, nout, outfunc, prior, beta=None):
    net_type = 'mlp'
    nwts = (nin+1)*nhidden + (nhidden+1)*nout
    net = Net(net_type, nin, nhidden, nout, nwts, outfunc)
    net.alpha = prior.alpha
    net.indx = prior.indx
    net.beta = beta
    
    net.w1 = np.random.randn(nin, nhidden)/np.sqrt(nin+1)
    net.b1 = np.random.randn(1, nhidden)/np.sqrt(nin+1)
    net.w2 = np.random.randn(nhidden, nout)/np.sqrt(nhidden+1)
    net.b2 = np.random.randn(1, nout)/np.sqrt(nhidden+1)
    return net
    
def MLP_init(net, prior):
    sig = 1 / np.sqrt(prior.indx.dot(prior.alpha))
    w = sig.T * (np.random.randn(1, net.nwts))
    net = MLP_unpack(net, w)
    return net

def MLP_unpack(net, w):
    if net.nwts != w.shape[1]:
        raise ValueError('Invalid weight vector length')
    
    nin = net.nin
    nhidden = net.nhidden
    nout = net.nout
    mark1 = nin*nhidden
    net.w1 = np.reshape(w[0][0:mark1], (nin, nhidden), order='F')
    mark2 = mark1 + nhidden
    net.b1 = np.reshape(w[0][mark1:mark2], (1, nhidden))
    mark3 = mark2 + nhidden*nout
    net.w2 = np.reshape(w[0][mark2:mark3], (nhidden, nout))
    mark4 = mark3 + nout
    net.b2 = np.reshape(w[0][mark3: mark4], (1, nout))
    return net

    
    
def MLP_fwd(net, xvals_t):
    ndata = xvals_t.shape[0]
    z = np.tanh(xvals_t.reshape(-1, 1).dot(net.w1) + np.ones((ndata, 1)).dot(net.b1))
    a = z.dot(net.w2) + np.ones((ndata, 1)).dot(net.b2)
    
    if net.outfunc == 'linear':
        y = a
    elif net.outfunc == 'logistic':
        maxcut = -np.log(np.finfo(float).eps)
        mincut = -np.log(1/np.finfo(float).tiny-1)
        a = min(a, maxcut)
        a = max(a, mincut)
        y = 1/(1 + np.exp(-a))
    elif net.outfunc == 'softmax':
        maxcut = np.log(float('inf'))-np.log(net.nout)
        mincut = np.log(np.finfo(float).tiny)
        a = min(a, maxcut)
        a = max(a, mincut)
        temp = np.exp(a)
        y = temp/(np.sum(temp, 1).dot(np.ones(1, net.nout)))
    else:
        raise ValueError('Unknown activation function')
        
    return y, a, z
        
    
    
params0 = np.array([5, 1, 1, 1])
params = np.tile(params0, (5, 1))
sf = 5

params[1, 0] = params0[0] * sf
params[2, 1] = params0[1] * sf
params[3, 2] = params0[2] * sf
params[4, 3] = params0[3] * sf

ntrials = 4

for t in range(ntrials):
    aw1 = 1/params[t, 0]**2
    aw2 = 1/params[t, 2]**2
    ab1 = 1/params[t, 1]**2
    ab2 = 1/params[t, 3]**2
    
    nhidden = 12
    nout = 1
    prior = MLPprior(1, nhidden, nout, aw1, ab1, aw2, ab2)
    xvals = np.arange(-1, 1.005, 0.005)
    nsample = 10
    net = MLP(1, nhidden, 1, 'linear', prior)
    
    fig = plt.figure(figsize=(22, 7))
    
    for i in range(nsample):
        net = MLP_init(net, prior)
        yvals, _, _ = MLP_fwd(net, xvals.T)
        plt.plot(xvals.T, yvals, color='k', lw=2)
        plt.title(r'$\sigma_1 = {},\; \tau_1 = {},\; \sigma_2 = {},\; \tau_2 = {}$'.format(1/np.sqrt(aw1),
                                                                                     1/np.sqrt(ab1),
                                                                                     1/np.sqrt(aw2),
                                                                                    1/np.sqrt(ab2)))
    
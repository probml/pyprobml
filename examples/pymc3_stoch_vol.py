#http://pymc-devs.github.io/pymc3/getting_started/#case-study-1-stochastic-volatility
#https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/stochastic_volatility.py

import numpy as np
import matplotlib.pyplot as plt
import pymc3
import pymc3.distributions.timeseries as ts
import pandas as pd
import scipy

#fname = 'https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/data/SP500.csv'
#returns = pd.read_csv(fname)
#returns = pd.read_csv('SP500.csv')
#print(len(returns))

n = 400
returns = np.genfromtxt(pymc3.get_data_file('pymc3.examples', "data/SP500.csv"))[-n:]
returns[:5]

plt.plot(returns)
plt.ylabel('daily returns in %');


with pymc3.Model() as sp500_model:
    nu = pymc3.Exponential('nu', 1./10, testval=5.)
    sigma = pymc3.Exponential('sigma', 1./.02, testval=.1)
    s = ts.GaussianRandomWalk('s', sigma**-2, shape=len(returns))
    volatility_process =  pymc3.Deterministic('volatility_process', pymc3.exp(-2*s))
    r = pymc3.StudentT('r', nu, lam=1/volatility_process, observed=returns)
    
with sp500_model:
    print 'optimizing...'
    start = pymc3.find_MAP(vars=[s], fmin=scipy.optimize.fmin_l_bfgs_b)
    
    print 'sampling... (slow!)'
    step = pymc3.NUTS(scaling=start)
    trace = pymc3.sample(100, step, progressbar=False)

    # Start next run at the last sampled position.
    step = pymc3.NUTS(scaling=trace[-1], gamma=.25)
    trace = pymc3.sample(1000, step, start=trace[-1], progressbar=False)
    

pymc3.traceplot(trace, [nu, sigma]);

fig, ax = plt.subplots(figsize=(15, 8))
#returns.plot(ax=ax)
ax.plot(returns)
ax.plot(1/np.exp(trace['s',::30].T), 'r', alpha=.03);
ax.set(title='volatility_process', xlabel='time', ylabel='volatility');
ax.legend(['S&P500', 'stochastic volatility process'])


# Robust Bayesian Binary logistic regression in 1d for iris flowers

# Code is based on 
# https://github.com/aloctavodia/BAP/blob/master/code/Chp4/04_Generalizing_linear_models.ipynb

import superimport

import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
#import seaborn as sns
import scipy.stats as stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import arviz as az
from sklearn.datasets import load_iris
import pyprobml_utils as pml

iris = load_iris()
X = iris.data 
y = iris.target

# Convert to pandas dataframe 
df_iris = pd.DataFrame(data=iris.data, 
                    columns=['sepal_length', 'sepal_width', 
                             'petal_length', 'petal_width'])
df_iris['species'] = pd.Series(iris.target_names[y], dtype='category')


df = df_iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length' 
x_0 = df[x_n].values
    
# Create outliers
x_outliers = np.array([4.2, 4.5, 4.0, 4.3, 4.2, 4.4])
y_outliers = np.ones_like(x_outliers, dtype=int)


Ninliers = len(x_0)
Noutliers = len(x_outliers)
N = Ninliers + Noutliers
inlier_ndx = np.arange(0, Ninliers)
outlier_ndx = np.arange(Ninliers, N)

y_0 = np.concatenate((y_0, y_outliers)) 
x_0 = np.concatenate((x_0, x_outliers))

xmean = np.mean(x_0)    
x_c = x_0 - xmean

def plot_training_data():
    plt.figure()
    for c in [0,1]:
        ndx_c = np.where(y_0==c)[0]
        color = f'C{c}'
        sigma = 0.02 # for vertical jittering
        inliers = np.intersect1d(ndx_c, inlier_ndx)
        plt.scatter(x_c[inliers], 
                    np.random.normal(y_0[inliers], sigma),
                    marker='o', color=color)
        outliers = np.intersect1d(ndx_c, outlier_ndx)
        plt.scatter(x_c[outliers], 
                    np.random.normal(y_0[outliers], sigma),
                    marker='x', color=color)
         
    plt.xlabel(x_n)
    plt.ylabel('p(y=1)', rotation=0)
    # use original scale for xticks
    locs, _ = plt.xticks()
    plt.xticks(locs, np.round(locs + xmean, 1))
    plt.tight_layout()



def infer_nonrobust_model():
    with pm.Model() as model_0:
        α = pm.Normal('α', mu=0, sd=10)
        β = pm.Normal('β', mu=0, sd=10)
        
        μ = α + pm.math.dot(x_c, β)    
        θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
        bd = pm.Deterministic('bd', -α/β) # decision boundary
        
        yl = pm.Bernoulli('yl', p=θ, observed=y_0)
    
        trace = pm.sample(1000, cores=1, chains=2)
        
    varnames = ['α', 'β', 'bd']
    az.summary(trace, varnames)
    return trace
        
def infer_robust_model():
    with pm.Model() as model_0:
        α = pm.Normal('α', mu=0, sd=10)
        β = pm.Normal('β', mu=0, sd=10)
        
        μ = α + pm.math.dot(x_c, β)    
        θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
        bd = pm.Deterministic('bd', -α/β) # decision boundary
        
        #yl = pm.Bernoulli('yl', p=θ, observed=y_0)
        π = pm.Beta('π', 1., 1.) # probability of contamination
        p = π * 0.5 + (1 - π) * θ # true prob or 0.5
        yl = pm.Bernoulli('yl', p=p, observed=y_0)
    
        trace = pm.sample(1000, cores=1, chains=2)
    
    varnames = ['α', 'β', 'bd', 'π']
    az.summary(trace, varnames)
    return trace
        


def make_plot(trace):       
    plot_training_data()
    # plot logistic curve
    theta = trace['θ'].mean(axis=0)
    idx = np.argsort(x_c)
    plt.plot(x_c[idx], theta[idx], color='C2', lw=3)
    az.plot_hdi(x_c, trace['θ'], color='C2')
    
    # plot decision boundary
    plt.vlines(trace['bd'].mean(), 0, 1, color='k')
    bd_hpd = az.hdi(trace['bd'])
    plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)
    

trace =  infer_robust_model()
make_plot(trace)     
pml.savefig('logreg_iris_bayes_robust_1d.pdf', dpi=300)

trace =  infer_nonrobust_model()
make_plot(trace)     
pml.savefig('logreg_iris_bayes_nonrobust_1d.pdf', dpi=300)

plt.show()

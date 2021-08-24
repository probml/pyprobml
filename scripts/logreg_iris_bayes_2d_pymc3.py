# Bayesian Binary logistic regression in 2d for iris flwoers

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

# We reduce the sample size from 50 to 25 per class,
# or to 5 + 45 in the unbalanced setting.
# The latter will increase posterior uncertainty
unbalanced = False # True
if unbalanced:
    df = df[45:95]
else:
    df = df[25:75]
assert(len(df)==50)

y_1 = pd.Categorical(df['species']).codes 
x_n = ['sepal_length', 'sepal_width'] 
x_1 = df[x_n].values


with pm.Model() as model_1: 
    α = pm.Normal('α', mu=0, sd=10) 
    β = pm.Normal('β', mu=0, sd=2, shape=len(x_n)) 
     
    μ = α + pm.math.dot(x_1, β) 
    θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ))) 
    bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * x_1[:,0])
     
    yl = pm.Bernoulli('yl', p=θ, observed=y_1) 
 
    trace_1 = pm.sample(2000, cores=1, chains=2)
    
varnames = ['α', 'β'] 
#az.plot_forest(trace_1, var_names=varnames);

idx = np.argsort(x_1[:,0]) 
bd = trace_1['bd'].mean(0)[idx] 

plt.figure()
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in y_1]) 
plt.plot(x_1[:,0][idx], bd, color='k'); 

az.plot_hdi(x_1[:,0], trace_1['bd'], color='k')
 
plt.xlabel(x_n[0]) 
plt.ylabel(x_n[1])

plt.tight_layout()
if unbalanced:
    pml.savefig('logreg_iris_bayes_2d_unbalanced.pdf', dpi=300)
else:
    pml.savefig('logreg_iris_bayes_2d.pdf', dpi=300)
        
plt.show()
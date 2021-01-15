# radon 1d linear regression with discrete covariates
# Source:
# Blog
# https://twiecki.io/blog/2014/03/17/bayesian-glms-3/
# Code:
# https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/GLM_hierarchical.ipynb

# Non-centered version
#https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/GLM_hierarchical_non_centered.ipynb

# Related links
#https://nbviewer.jupyter.org/github/fonnesbeck/multilevel_modeling/blob/master/multilevel_modeling.ipynb?create=1

# PyMC4 version
#https://github.com/pymc-devs/pymc4/blob/master/notebooks/radon_hierarchical.ipynb


#matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm 
import pandas as pd

data = pd.read_csv('radon.csv')

county_names = data.county.unique()
county_idx = data['county_code'].values

data[['county', 'log_radon', 'floor']].head()


#http://pymc-devs.github.io/pymc3/getting_started/#generalized-linear-models
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import *
import pandas as pd
import scipy

N = 100;
X1 = np.random.randn(N,1)
X2 = np.random.randn(N,1)
Y = np.random.randn(N,1)
df = pd.DataFrame({'x1': X1, 'x2': X2, 'y': Y}, index=range(N))

with Model() as model_glm:
    glm('y ~ x1 + x2', df)
    trace = sample(5000)
    
traceplot(trace);

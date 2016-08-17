# linear regression for prostate cancer dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.cross_validation as cv
import sklearn.linear_model as lm
import scipy.io
import patsy


# Prevent numpy from printing too many digits
np.set_printoptions(precision=3)

# Prevent Pandas from printing results in html format
pd.set_option('notebook_repr_html', False)
pd.set_option('display.max_columns', 160)
pd.set_option('display.width', 1000)

# Get data
url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)
# skip the column of indices
df = df.drop('Unnamed: 0', axis=1)

# Convert the final column of T/F into boolean
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)

# Scale the X variables
scaler = sklearn.preprocessing.StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']  # restore the unscaled y column

##############################

# Helper functions
def L2loss(yhat, ytest):
    ntest = ytest.size
    sqerr = np.power(yhat - ytest, 2)
    mse = np.mean(sqerr)
    stderr = np.std(sqerr) / np.sqrt(ntest)
    return (mse, stderr)

def fitAndPredict(X, y, train_mask):
    #  partition into train/test set
    Xtrain = X[train_mask]
    ytrain = y[train_mask]
    Xtest = X[np.logical_not(train_mask)]
    ytest = y[np.logical_not(train_mask)]
    
    # Fit model
    linreg = lm.LinearRegression(fit_intercept = False)
    linreg.fit(Xtrain, ytrain)
    
    # Extract parameters
    coef = linreg.coef_
    names = X.columns.tolist()
    print([name + ':' + str(round(w,3)) for name, w in zip(names, coef)])
    
    # Measure train error
    yhat = linreg.predict(Xtrain)
    (mse, stderr) = L2loss(yhat, ytrain) 
    print 'Train mse {0:0.3f}, stderr {1:0.3f}'.format(mse, stderr) 
    
    # Measure test error
    yhat = linreg.predict(Xtest)
    (mse, stderr) = L2loss(yhat, ytest) 
    print 'Test mse {0:0.3f}, stderr {1:0.3f}'.format(mse, stderr) 


##############################

# use all the variables, without dummy encoding. Adds column of 1s.
#y, X = patsy.dmatrices('lpsa ~ .', df, return_type="dataframe")
#formula = 'lpsa ~ {0}'.format(' + '.join(xcols))
#y, X = patsy.dmatrices(formula, df_scaled, return_type="dataframe")
#y = np.ravel(y)

X = df_scaled.ix[:,:-1] # exclude final lpsa column 
N = X.shape[0]
X.insert(0, 'intercept', np.ones(N))
y = df_scaled['lpsa'] # pandas series

'''
   intercept    lcavol   lweight       age      lbph       svi       lcp   gleason     pgg45
0          1 -1.645861 -2.016634 -1.872101 -1.030029 -0.525657 -0.867655 -1.047571 -0.868957
1          1 -1.999313 -0.725759 -0.791989 -1.030029 -0.525657 -0.867655 -1.047571 -0.868957
2          1 -1.587021 -2.200154  1.368234 -1.030029 -0.525657 -0.867655  0.344407 -0.156155
3          1 -2.178174 -0.812191 -0.791989 -1.030029 -0.525657 -0.867655 -1.047571 -0.868957
4          1 -0.510513 -0.461218 -0.251933 -1.030029 -0.525657 -0.867655 -1.047571 -0.868957
'''

print('vanilla model on scaled X')
fitAndPredict(X, y, istrain)

# Matches HTF book
#['intercept:2.465', 'lcavol:0.676', 'lweight:0.262', 'age:-0.141', 'lbph:0.209', 
#'svi:0.304', 'lcp:-0.287', 'gleason:-0.021', 'pgg45:0.266']

#Train mse 0.439, stderr 0.079
#Test mse 0.521, stderr 0.176




##############################

# Treat categorical variables properly
df_cat = df_scaled.copy()
# svi has values {0,1}
df_cat["svi"] = df["svi"].astype('category')
# gleason has  values (6,7,8,9), so we convert to factor
df_cat["gleason"] = df["gleason"].astype('category')

# Look at the reformatted data. Notice how lpsa, svi and gleason match
# original data, whereas other columns are scaled
df_cat.head()

#     lcavol   lweight       age      lbph svi       lcp gleason     pgg45      lpsa
#0 -1.645861 -2.016634 -1.872101 -1.030029   0 -0.867655       6 -0.868957 -0.430783
#1 -1.999313 -0.725759 -0.791989 -1.030029   0 -0.867655       6 -0.868957 -0.162519
#2 -1.587021 -2.200154  1.368234 -1.030029   0 -0.867655       7 -0.156155 -0.162519
#3 -2.178174 -0.812191 -0.791989 -1.030029   0 -0.867655       6 -0.868957 -0.162519
#4 -0.510513 -0.461218 -0.251933 -1.030029   0 -0.867655       6 -0.868957  0.371564

# create summary of numeric and categorical columns
df_cat.describe(include='all')


##############################

# Basic model. Adds column of 1s at start.
X = patsy.dmatrix('lcavol + lweight + age + lbph + lcp + svi + gleason + pgg45',
    df_cat, return_type="dataframe")
y = df_cat['lpsa']

X.head()

#   Intercept  svi[T.1]  gleason[T.7]  gleason[T.8]  gleason[T.9]    lcavol   lweight       age      lbph       lcp     pgg45
#0          1         0             0             0             0 -1.645861 -2.016634 -1.872101 -1.030029 -0.867655 -0.868957
#1          1         0             0             0             0 -1.999313 -0.725759 -0.791989 -1.030029 -0.867655 -0.868957
#2          1         0             1             0             0 -1.587021 -2.200154  1.368234 -1.030029 -0.867655 -0.156155
#3          1         0             0             0             0 -2.178174 -0.812191 -0.791989 -1.030029 -0.867655 -0.868957
#4          1         0             0             0             0 -0.510513 -0.461218 -0.251933 -1.030029 -0.867655 -0.868957

print('categorical')
fitAndPredict(X, y, istrain)

#['Intercept:2.218', 'svi[T.1]:0.707', 'gleason[T.7]:0.183', 'gleason[T.8]:0.727', 'gleason[T.9]:-0.497', 
#'lcavol:0.666', 'lweight:0.275', 'age:-0.164', 'lbph:0.19', 'lcp:-0.341', 'pgg45:0.31']

#Train mse 0.413, stderr 0.070
#Test mse 0.551, stderr 0.179


##############################

# Rename columns to make them shorter (for printing purposes)
df_cat = df_cat.rename(columns = {'gleason':'g', 'svi':'s'})

# Interaction terms.
X = patsy.dmatrix('lcavol + lweight + age + lbph + lcp + s:g + pgg45',
    df_cat, return_type="dataframe")
y = df_cat['lpsa']
                        
X.head()

'''
   Intercept  g[T.7]  g[T.8]  g[T.9]  s[T.1]:g[6]  s[T.1]:g[7]  s[T.1]:g[8]  s[T.1]:g[9]    lcavol   lweight       age      lbph       lcp     pgg45
0          1       0       0       0            0            0            0            0 -1.645861 -2.016634 -1.872101 -1.030029 -0.867655 -0.868957
1          1       0       0       0            0            0            0            0 -1.999313 -0.725759 -0.791989 -1.030029 -0.867655 -0.868957
2          1       1       0       0            0            0            0            0 -1.587021 -2.200154  1.368234 -1.030029 -0.867655 -0.156155
3          1       0       0       0            0            0            0            0 -2.178174 -0.812191 -0.791989 -1.030029 -0.867655 -0.868957
4          1       0       0       0            0            0            0            0 -0.510513 -0.461218 -0.251933 -1.030029 -0.867655 -0.868957
'''

print('interaction ')
fitAndPredict(X, y, istrain)



#['Intercept:2.223', 'gleason[T.7]:0.121', 'gleason[T.8]:0.641', 'gleason[T.9]:0.048', 
#'svi[T.1]:gleason[6]:0.0', 'svi[T.1]:gleason[7]:0.849', 'svi[T.1]:gleason[8]:0.0', 'svi[T.1]:gleason[9]:-0.872', 
#'lcavol:0.631', 'lweight:0.265', 'age:-0.135', 'lbph:0.243', 'lcp:-0.292', 'pgg45:0.287']

#Train mse 0.390, stderr 0.075
#Test mse 0.575, stderr 0.170



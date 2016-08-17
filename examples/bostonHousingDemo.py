# Boston housing demo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.cross_validation as cv
import sklearn.linear_model as lm
import scipy.io


# Prevent numpy from printing too many digits
np.set_printoptions(precision=3)

# ensure reproducibility
np.random.seed(42) 

# Prevent Pandas from printing results in html format
pd.set_option('notebook_repr_html', False)


# Load data (creates numpy arrays)
boston = sklearn.datasets.load_boston()
X = boston.data
y = boston.target

# Save to matlab format for later use
#fname = '/Users/kpmurphy/github/pmtkdata/bostonHousing/boston.mat'
#dict = {'X':X, 'y':y, 'names':boston.feature_names};
#scipy.io.savemat(fname, dict)

# Convert to Pandas format
df = pd.DataFrame(X)
df.columns = boston.feature_names
df['MEDV'] = y.tolist()

df.describe()
'''
             CRIM          ZN       INDUS        CHAS         NOX          RM 
count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   
mean     3.593761   11.363636   11.136779    0.069170    0.554695    6.284634   
std      8.596783   23.322453    6.860353    0.253994    0.115878    0.702617   
min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000   
25%      0.082045    0.000000    5.190000    0.000000    0.449000    5.885500   
50%      0.256510    0.000000    9.690000    0.000000    0.538000    6.208500   
75%      3.647423   12.500000   18.100000    0.000000    0.624000    6.623500   
max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000   

              AGE         DIS         RAD         TAX     PTRATIO           B  
count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   
mean    68.574901    3.795043    9.549407  408.237154   18.455534  356.674032   
std     28.148861    2.105710    8.707259  168.537116    2.164946   91.294864   
min      2.900000    1.129600    1.000000  187.000000   12.600000    0.320000   
25%     45.025000    2.100175    4.000000  279.000000   17.400000  375.377500   
50%     77.500000    3.207450    5.000000  330.000000   19.050000  391.440000   
75%     94.075000    5.188425   24.000000  666.000000   20.200000  396.225000   
max    100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   

            LSTAT        MEDV  
count  506.000000  506.000000  
mean    12.653063   22.532806  
std      7.141062    9.197104  
min      1.730000    5.000000  
25%      6.950000   17.025000  
50%     11.360000   21.200000  
75%     16.955000   25.000000  
max     37.970000   50.000000
'''

# plot marginal histograms of each column (13 features, 1 response)
df.hist()
plt.show()

'''
# plot marginal histograms of each column (13 features, 1 response)
nrows = 7; ncols = 2;
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[15,10])
plt.tight_layout()
plt.clf()
for i in range(0,13):
    subplot(nrows, ncols, i+1)
    hist(X[:,i])
    title(boston.feature_names[i])
    plt.locator_params(axis = 'y', nbins = 3) # fewer yticks
subplot(nrows, ncols, 14)
hist(y)
title('Median price ($1000 USD)')
plt.show()
plt.locator_params(axis = 'y', nbins = 3)
'''

# scatter plot of response vs each feature 
nrows = 3; ncols = 4;
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=[15, 10])
plt.tight_layout()
plt.clf()
for i in range(0,12):
    plt.subplot(nrows, ncols, i+1)
    plt.scatter(X[:,i], y)
    plt.title(boston.feature_names[i])
    plt.locator_params(axis = 'y', nbins = 3)
    plt.locator_params(axis = 'x', nbins = 5)
    plt.grid()
plt.show()

# Rescale input data
scaler = sklearn.preprocessing.StandardScaler()
scaler = scaler.fit(X)
X = scaler.transform(X)

# Fit model
linreg = lm.LinearRegression()
linreg.fit(X, y)

# Extract parameters
coef = np.append(linreg.coef_, linreg.intercept_)
names = np.append(boston.feature_names, 'intercept')
print([name + ':' + str(round(w,1)) for name, w in zip(names, coef)])

# Assess fit on training set
yhat = linreg.predict(X) 
plt.figure()
plt.scatter(y, yhat)
plt.xlabel("true price")
plt.ylabel("predicted price")
plt.title("Predicted vs true house price (x $1000 USD) for Boston, 1978")
xs = np.linspace(min(y), max(y), 100)
plt.plot(xs, xs, '-')
plt.show()

print "R^2 on training set is {0:0.3f}".format(linreg.score(X,y))

# evaluate the model using 10-fold cross-validation
scores = cv.cross_val_score(lm.LinearRegression(), 
            X, y, scoring='r2', cv=10)
print scores
print "Median R^2 across CV folds is {0:0.3f}".format(np.median(scores))

# Direct implementation of OLS (unfinished)
#N = X.shape[0]
#X1 = np.append(X, np.ones(N))

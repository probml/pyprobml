# logistic regression demo on college entrance data 
# Based on http://blog.yhat.com/posts/logistic-regression-and-python.html
# But we modified it to use scikit-learn intsead of statsmodels for fitting.
# We also borrowed some code from 
# https://github.com/justmarkham/gadsdc1/blob/master/logistic_assignment/kevin_logistic_sklearn.ipynb
# For details on how to do this analysis in R, see http://www.ats.ucla.edu/stat/r/dae/logit.htm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics

# Prevent Pandas from printing results in html format
pd.set_option('notebook_repr_html', False)

# Prevent numpy from printing too many digits
np.set_printoptions(precision=3)

# read the data in
df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
#df.to_csv('/Users/kpmurphy/github/pmtk3/data/collegeAdmissions.csv')


# rename the 'rank' column because there is also a DataFrame method called 'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
print df.columns
# array([admit, gre, gpa, prestige], dtype=object)


# take a look at the dataset
print df.head()
#    admit  gre   gpa  prestige
# 0      0  380  3.61     3
# 1      1  660  3.67     3
# 2      1  800  4.00     1
# 3      1  640  3.19     4
# 4      0  520  2.93     4



# print summary statistics
df.describe()
#            admit         gre         gpa       prestige
#count  400.000000  400.000000  400.000000  400.00000
#mean     0.317500  587.700000    3.389900    2.48500
#std      0.466087  115.516536    0.380567    0.94446
#min      0.000000  220.000000    2.260000    1.00000
#25%      0.000000  520.000000    3.130000    2.00000
#50%      0.000000  580.000000    3.395000    2.00000
#75%      1.000000  660.000000    3.670000    3.00000
#max      1.000000  800.000000    4.000000    4.00000


# plot all of the columns
df.hist()
plt.show()

df.groupby('prestige').mean()

#             admit         gre       gpa
#prestige                                
#1         0.540984  611.803279  3.453115
#2         0.357616  596.026490  3.361656
#3         0.231405  574.876033  3.432893
#4         0.179104  570.149254  3.318358

#######
# Explicitly deal with dummy encoding

# convert prestige column to dummy encoding
dummy = pd.get_dummies(df['prestige'], prefix='prestige')
print dummy.head()
#    prestige_1  prestige_2  prestige_3  prestige_4
# 0           0           0           1           0
# 1           0           0           1           0
# 2           1           0           0           0
# 3           0           0           0           1
# 4           0           0           0           1

# create a clean data frame with the variables of interest
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy.ix[:, 'prestige_2':])

# manually add the intercept
data['intercept'] = 1.0

print data.head()
#   admit  gre   gpa  prestige_2  prestige_3  prestige_4  intercept
#0      0  380  3.61           0           1           0          1
#1      1  660  3.67           0           1           0          1
#2      1  800  4.00           0           0           0          1
#3      1  640  3.19           0           0           1          1
#4      0  520  2.93           0           0           1          1


# StasModels version
train_cols = data.columns[1:]
logit = sm.Logit(data['admit'], data[train_cols])
result = logit.fit()
coef_sm = result.params
print coef_sm

#########
# Let's use patsy notation to handle dummy variables.
# This adds a column of 1s
y, X = dmatrices('admit ~ gre + gpa + C(prestige)', df, return_type="dataframe")
y = np.ravel(y)
X.head()
#  Intercept  C(prestige)[T.2]  C(prestige)[T.3]  C(prestige)[T.4]  gre   gpa
#0          1                 0                 1                 0  380  3.61
#1          1                 0                 1                 0  660  3.67
#2          1                 0                 0                 0  800  4.00
#3          1                 0                 0                 1  640  3.19
#4   

# Now we switch to scikit learn
# We set the inverse regularizer, C, to infinity to make sure we're doing MLE
#http://stackoverflow.com/questions/24924755/logit-estimator-in-statsmodels-and-sklearn
                            
model = LogisticRegression(fit_intercept=False, C=1e9)
y = np.ravel(y)
model = model.fit(X, y)
coef_patsy = np.ravel(model.coef_)
pd.DataFrame(zip(X.columns, [round(c, 3) for c in coef_patsy]))

#
# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(fit_intercept=False, C=1e9), 
            X, y, scoring='accuracy', cv=10)
print scores
#[ 0.805  0.61   0.732  0.725  0.675  0.7    0.7    0.692  0.744  0.667]

print 'average CV accuracy = {0:.2f}'.format(scores.mean()) #0.70
print 'baseline accuracy = {0:.2f}'.format(1-y.mean()) # 0.68
   
   
# Split data into train and test sets.
# We first shuffle the order of the rows (although this is done inside the
# train_test_split function, so is not strictly neccessary).
np.random.seed(42) # ensure reproducibility
#X = pd.DataFrame(np.random.randn(5,2))
#y = np.random.rand(5)
N = len(X)
ndx = np.random.permutation(range(N))
Xshuffled = X.reindex(ndx) # dataframe
yshuffled = y[ndx] # numpy 1d array
X_train, X_test, y_train, y_test = train_test_split(Xshuffled, yshuffled, test_size=0.3, random_state=0)

# fit model
model = model.fit(X_train, y_train)

# predictions on test set
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

# check the accuracy
print 'baseline accuracy = {0:.2f}'.format(1-y.mean())
print 'accuracy on test set = {0:.2f}'.format(metrics.accuracy_score(y_test, predicted))
print 'auc on test set = {0:.2f}'.format(metrics.roc_auc_score(y_test, probs[:, 1]))
print 'class confusion matrix'
print metrics.confusion_matrix(y_test, predicted)



                                    
####
# Now let's work with numpy instead of dataframes
'''
# Convert dummy df to numpy array. We drop the column of 1s
X = data.ix[:, data.columns[1:-1]].as_matrix()
y = data.ix[:, 'admit'].as_matrix()

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression(C=1e9)
model = model.fit(X, y)
coef_np = np.append(model.coef_, model.intercept_)
names = np.append(data.columns[1:-1].values, 'intercept')
pd.DataFrame(zip(names, [round(c, 3) for c in coef_np]))
'''

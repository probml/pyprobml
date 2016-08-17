#logistic regression demo on female extra marital affair dataset
#https://github.com/justmarkham/gadsdc1/blob/master/logistic_assignment/kevin_logistic_sklearn.ipynb


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# load dataset
data = sm.datasets.fair.load_pandas().data

# add "affair" column: 1 represents having affairs, 0 represents not
data['affair'] = (data.affairs > 0).astype(int)

# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
# We use the patsy package to create the design matrix
# http://patsy.readthedocs.org/en/latest/overview.html
# This encodes categorical variables by dropping the first level

y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
                  religious + educ + C(occupation) + C(occupation_husb)',
                  data, return_type="dataframe")
                   
# fix column names of X 
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})
                        
# flatten y into a 1-D array
y = np.ravel(y)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X_train, y_train)

# examine the coefficients
coef = np.ravel(model.coef_)
coef_round = [round(c, 2) for c in coef]
pd.DataFrame(zip(X.columns, np.transpose(coef_round)))

# predictions on test set
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

# check the accuracy
print 'baseline accuracy = {0}'.format(1-y.mean())
print 'accuracy on test set = {0}'.format(metrics.accuracy_score(y_test, predicted))
print 'auc on test set = {0}'.format(metrics.roc_auc_score(y_test, probs[:, 1]))
print 'class confusion matrix'
print metrics.confusion_matrix(y_test, predicted)





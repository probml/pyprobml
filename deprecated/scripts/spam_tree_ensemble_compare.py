

#Performance of tree ensembles. Based on the email spam example from chapter 10 of "Elements of statistical learning". Code is from Andrey Gaskov's site:

#https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/examples/Spam.ipynb


# Commented out IPython magic to ensure Python compatibility.
import superimport

import pandas as pd
from matplotlib import transforms, pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# omit numpy warnings (don't do it in real work)
np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore')
# %matplotlib inline

# define plots common properties and color constants
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
ORANGE, BLUE, PURPLE = '#FF8C00', '#0000FF', '#A020F0'
GRAY1, GRAY4, GRAY7 = '#231F20', '#646369', '#929497'


# we will calculate train and test error rates for all models
def error_rate(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

"""Get data"""

df = pd.read_csv("https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/data/Spam.txt?raw=True")
df.head()

# PAGE 301. We coded spam as 1 and email as zero. A test set of size 1536 was
#           randomly chosen, leaving 3065 observations in the training set.
target = 'spam'
columns = ['word_freq_make', 'word_freq_address', 'word_freq_all',
           'word_freq_3d', 'word_freq_our', 'word_freq_over',
           'word_freq_remove', 'word_freq_internet', 'word_freq_order',
           'word_freq_mail', 'word_freq_receive', 'word_freq_will',
           'word_freq_people', 'word_freq_report', 'word_freq_addresses',
           'word_freq_free', 'word_freq_business', 'word_freq_email',
           'word_freq_you', 'word_freq_credit', 'word_freq_your',
           'word_freq_font', 'word_freq_000', 'word_freq_money',
           'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
           'word_freq_650', 'word_freq_lab', 'word_freq_labs',
           'word_freq_telnet', 'word_freq_857', 'word_freq_data',
           'word_freq_415', 'word_freq_85', 'word_freq_technology',
           'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
           'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
           'word_freq_original', 'word_freq_project', 'word_freq_re',
           'word_freq_edu', 'word_freq_table', 'word_freq_conference',
           'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
           'char_freq_$', 'char_freq_#', 'capital_run_length_average',
           'capital_run_length_longest', 'capital_run_length_total']
# let's give columns more compact names
features = ['make', 'address', 'all', '3d', 'our', 'over', 'remove',
            'internet', 'order', 'mail', 'receive', 'will', 'people',
            'report', 'addresses', 'free', 'business', 'email', 'you',
            'credit', 'your', 'font', '000', 'money', 'hp', 'hpl',
            'george', '650', 'lab', 'labs', 'telnet', '857', 'data',
            '415', '85', 'technology', '1999', 'parts', 'pm', 'direct',
            'cs', 'meeting', 'original', 'project', 're', 'edu', 'table',
            'conference', 'ch_;', 'ch(', 'ch[', 'ch!', 'ch$', 'ch#',
            'CAPAVE', 'CAPMAX', 'CAPTOT']

X, y = df[columns].values, df[target].values

# split by test column value
is_test = df.test.values
X_train, X_test = X[is_test == 0], X[is_test == 1]
y_train, y_test = y[is_test == 0], y[is_test == 1]

""" Logistic regression

As a sanity check, we try to match p301  test error rate of 7.6%.

"""

import statsmodels.api as sm
from sklearn.metrics import accuracy_score

lr_clf = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=False)
# 0.5 is a threshold
y_test_hat = (lr_clf.predict(sm.add_constant(X_test)) > 0.5).astype(int)
lr_error_rate = error_rate(y_test, y_test_hat)
print(f'Logistic Regression Test Error Rate: {lr_error_rate*100:.1f}%')





# PAGE 590. A random forest classifier achieves 4.88% misclassification error
#           on the spam test data, which compares well with all other methods,
#           and is not significantly worse than gradient boosting at 4.5%.
ntrees_list = [10, 50, 100, 200, 300, 400, 500]



from sklearn.ensemble import RandomForestClassifier
rf_errors = []
for ntrees in ntrees_list:
    rf_clf = RandomForestClassifier(
        n_estimators=ntrees,
        random_state=10
    ).fit(X_train, y_train)
    y_test_hat = rf_clf.predict(X_test)
    rf_error_rate = error_rate(y_test, y_test_hat)
    rf_errors.append(rf_error_rate)
    print(f'RF {ntrees} trees, test error {rf_error_rate*100:.1f}%')

from catboost import CatBoostClassifier, Pool, cv

boost_errors = []
for ntrees in ntrees_list:
    boost_clf = CatBoostClassifier(
        iterations=ntrees,
        random_state=10,
        learning_rate=0.2,
        verbose=False
    ).fit(X_train, y_train)
    y_test_hat = boost_clf.predict(X_test)
    boost_error_rate = error_rate(y_test, y_test_hat)
    boost_errors.append(boost_error_rate)
    print(f'Boosting {ntrees} trees, test error {boost_error_rate*100:.1f}%')
    
from sklearn.ensemble import BaggingClassifier
bag_errors = []
for ntrees in ntrees_list:
    bag_clf = BaggingClassifier(
        n_estimators=ntrees,
        random_state=10,
        bootstrap=True
    ).fit(X_train, y_train)
    y_test_hat = bag_clf.predict(X_test)
    bag_error_rate = error_rate(y_test, y_test_hat)
    bag_errors.append(bag_error_rate)
    print(f'Bagged {ntrees} trees, test error {bag_error_rate*100:.1f}%')
    

plt.figure()
plt.plot(ntrees_list, rf_errors, 'o-', color='blue', label='RF')
plt.plot(ntrees_list, boost_errors, 'x-', color='green', label='Boosting')
plt.plot(ntrees_list, bag_errors, '^-', color='orange', label='Bagging')
plt.legend()
plt.xlabel('Number of trees')
plt.ylabel('Test error')
plt.tight_layout()
plt.savefig('../figures/spam_tree_ensemble_compare.pdf', dpi=300)






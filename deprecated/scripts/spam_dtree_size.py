

#Performance of tree ensembles. Based on the email spam example from chapter 10 of "Elements of statistical learning". Code is from Andrey Gaskov's site:

#https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/examples/Spam.ipynb

import superimport

from one_standard_error_rule_model import OneStandardErrorRuleModel
from sklearn import tree

# Commented out IPython magic to ensure Python compatibility.
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


#max_leaf_nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 21, 26, 30, 33, 37, 42]
max_leaf_nodes = [int(x) for x in np.linspace(2,200,10)]

tree_based_clf = OneStandardErrorRuleModel(
    tree.DecisionTreeClassifier(criterion='entropy', random_state=5),
    'max_leaf_nodes', max_leaf_nodes,
    is_regression=False, random_state=26,
).fit(X_train, y_train)
print(f'Selected max_leaf_nodes: {tree_based_clf.model_.max_leaf_nodes}')
print(f'Test error rate: {tree_based_clf.assess(X_test, y_test)[0]*100:.1f}%')

# calculate test error rate for each parameter value
test_error_rates = [
    tree_based_clf.refit(X_train, y_train, i).assess(X_test, y_test)[0]
    for i in range(len(max_leaf_nodes))]

# PAGE 313. Figure 9.4 shows the 10-fold cross-validation error rate as a
#           function of the size of the pruned tree, along with ±2 standard
#           errors of the mean, from the ten replications. The test error curve
#           is shown in orange.
fig, ax = plt.subplots(figsize=(4.75, 3.15), dpi=150)
ax.plot(max_leaf_nodes, tree_based_clf.cv_mean_errors_, c=BLUE, linewidth=0.6)
ax.errorbar(max_leaf_nodes, tree_based_clf.cv_mean_errors_,
            color=BLUE, linestyle='None', marker='o', elinewidth=0.2,
            markersize=1.5, yerr=tree_based_clf.cv_mean_errors_std_,
            ecolor=BLUE, capsize=2)
ax.axhline(y=tree_based_clf.cv_min_error_ + tree_based_clf.cv_min_error_std_,
           c=GRAY1, linewidth=0.6, linestyle=':')
for e in ax.get_yticklabels() + ax.get_xticklabels():
    e.set_fontsize(6)
ax.set_xlabel('Tree size', color=GRAY4, fontsize=7)
ax.set_ylabel('Misclassification Rate', color=GRAY4, fontsize=7)
ax.scatter(max_leaf_nodes, test_error_rates, color=ORANGE,
           s=3, zorder=10)
ax.plot(max_leaf_nodes, test_error_rates, color=ORANGE,
        linewidth=0.6)
_ = ax.set_ylim(-0.02, 0.47)
plt.tight_layout()


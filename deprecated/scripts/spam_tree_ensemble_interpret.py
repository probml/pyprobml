


import superimport

#Feature importance using tree ensembles. Based on the email spam example from chapter 10 of "Elements of statistical learning". Code is from Andrey Gaskov's site:

#https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/examples/Spam.ipynb


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

"""Boosting

p353 says gradient boosted trees (J=5 leaves) has test error is 4.5%.
"""

#!pip install catboost

from catboost import CatBoostClassifier, Pool, cv

# find the best number of trees using 5 fold cross validation
params = {
    'loss_function': 'Logloss',
    'iterations': 100,
    'eval_metric': 'Accuracy',
    'random_seed': 100,
    'learning_rate': 0.2}
cv_data = cv(
    params=params,
    pool=Pool(X_train, label=y_train),
    fold_count=5,
    shuffle=True,
    partition_random_seed=0,
    stratified=True,
    verbose=False
)
best_iter = np.argmax(cv_data['test-Accuracy-mean'].values)
print(f'Best number of iterations {best_iter}')

# PAGE 352. Applying gradient boosting to these data resulted in a test error
#           rate of 4.5%, using the same test set as was used in Section 9.1.2.

# refit model the the whole data using found the best number of trees
cb_clf = CatBoostClassifier(
    iterations=best_iter,
    random_seed=100,
    learning_rate=0.2
).fit(X_train, y_train, verbose=False)
cb_error_rate = error_rate(y_test, cb_clf.predict(X_test))
print(f'Boosting Test Error Rate: {cb_error_rate*100:.1f}%')

# Feature importance

def plot_relative_feature_importance(importance):
    max_importance = np.max(importance)
    relative_importance = sorted(zip(100*importance/max_importance, features))
    yticks = np.arange(len(relative_importance))
    yticklabels = [ri[1] for ri in relative_importance][::-1]
    bars_sizes = [ri[0] for ri in relative_importance][::-1]

    fig, ax = plt.subplots(figsize=(4.3, 6.5), dpi=150)
    bars = ax.barh(yticks, bars_sizes, height=0.8, color='red')
    plt.setp(ax, yticks=yticks, yticklabels=yticklabels)
    ax.set_xlim([0, 100])
    ax.set_ylim([-0.5, 57])
    for e in ax.get_yticklabels()+ax.get_xticklabels():
        e.set_fontsize(6)
        e.set_color(GRAY1)
    ax.tick_params(left=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    offset = transforms.ScaledTranslation(0, -0.07, fig.dpi_scale_trans)
    for e in ax.get_xticklabels() + ax.xaxis.get_ticklines() + \
             [ax.spines['bottom']]:
        e.set_transform(e.get_transform() + offset)
    ax.spines['bottom'].set_bounds(0, 100)
    _ = ax.set_xlabel('Relative Importance', color=GRAY4, fontsize=7)

# PAGE 354. FIGURE 10.6. Predictor variable importance spectrum for the spam
#           data. The variable names are written on the vertical axis.
plot_relative_feature_importance(np.array(cb_clf.get_feature_importance()))
plt.tight_layout()
plt.savefig('../figures/spam_feature_importance.pdf', dpi=300)
plt.show()

# Partial dependence 

def plot_partial_dependence(ax, feature):
    n = features.index(feature)
    X_tmp = X.copy()
    vals = np.unique(np.percentile(X_tmp[:, n], np.linspace(5, 95, 100)))
    result = []
    for i in range(vals.shape[0]):
        X_tmp[:, n] = vals[i]
        pr = np.mean(cb_clf.predict_proba(X_tmp), axis=0)
        result.append(np.log(pr[1]/pr[0]))
    #ax.plot(vals, result, linewidth=0.6, color='#26FF26')
    ax.plot(vals, result, linewidth=1.5, color='#26FF26')
    for e in ax.get_yticklabels() + ax.get_xticklabels():
        e.set_fontsize(4)
    ax.set_ylabel('Partial Dependance', color=GRAY4, fontsize=8)
    ax.set_xlabel(f'{feature}', color=GRAY4, fontsize=8)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.15)

    # plot small red lines for the data deciles
    deciles = np.percentile(X[:, n], np.linspace(10, 90, 9))
    y_from, y_to = ax.get_ylim()
    for i in range(deciles.shape[0]):
        x = deciles[i]
        ax.plot([x, x], [y_from, y_from+(y_to-y_from)*0.05],
                color='red', linewidth=1)
    ax.set_ylim(y_from, y_to)

# PAGE 355. FIGURE 10.7. Partial dependence of log-odds of spam on four
#           important predictors. The red ticks at the base of the plots are
#           deciles of the input variable.
fig, axarr = plt.subplots(2, 2, figsize=(4.65, 3.43), dpi=150)
plt.subplots_adjust(wspace=0.35, hspace=0.45)
plot_partial_dependence(axarr[0, 0], 'ch!')
plot_partial_dependence(axarr[0, 1], 'remove')
plot_partial_dependence(axarr[1, 0], 'edu')
plot_partial_dependence(axarr[1, 1], 'hp')
plt.tight_layout()
plt.savefig('../figures/spam_partial_dependence.pdf', dpi=300)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# prepare colormap the looks similar to colormap from the book
N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1, 0, N)
vals[:, 1] = np.linspace(0, 1, N)
vals[:, 2] = np.linspace(1, 1, N)
newcmp = ListedColormap(vals)

# calculate coordinates grids for surface and frame plotting
n1, n2 = features.index('hp'), features.index('ch!')
vals1, vals2 = np.linspace(0, 3, 40), np.linspace(0, 1, 20)
N1, N2 = np.meshgrid(vals1, vals2)
LO = np.zeros(shape=N1.shape)

X_tmp = X.copy()
for i in range(N1.shape[0]):
    for j in range(N1.shape[1]):
        X_tmp[:, n1], X_tmp[:, n2] = N1[i, j], N2[i, j]
        pr = np.mean(cb_clf.predict_proba(X_tmp), axis=0)
        LO[i, j] = np.log(pr[1]/pr[0])

# PAGE 355. FIGURE 10.8. Partial dependence of the log-odds of spam vs. email
#            as a function of joint frequences of hp and the character !.
fig = plt.figure(figsize=(6, 3.75), dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xlabel('hp', fontsize=8)
ax.set_ylabel('ch!', fontsize=8)
ax.w_xaxis.line.set_color(GRAY7)
ax.w_yaxis.line.set_color(GRAY7)
ax.w_zaxis.line.set_color(GRAY7)
ax.view_init(22, 81)
# invert y-axis
ax.set_ylim(1, 0)
ax.set_xlim(0, 3)
for e in ax.get_yticklabels() + ax.get_xticklabels() + \
         ax.get_zticklabels():
    e.set_fontsize(7)

ax.plot_surface(N1, N2, LO, cmap=newcmp, shade=False)
_ = ax.plot_wireframe(N1, N2, LO, cmap=newcmp, linewidth=0.5, color='black')
plt.tight_layout()
plt.savefig('../figures/spam_joint_partial_dependence.pdf', dpi=300)
plt.show()



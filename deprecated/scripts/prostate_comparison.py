'''
Compares L1, L2, allSubsets, and OLS linear regression on the prostate data set
Author : Aleyna Kara (@karalleyna)
Based on https://github.com/probml/pmtk3/blob/master/demos/prostateComparison.m
Sourced from https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/examples/Prostate%20Cancer.ipynb
'''

import superimport

import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def get_features_and_label(dataset, is_training):
    '''
    Gets matrices representing features and target from the original data set
    Parameters
    ----------
    dataset: DataFrame
      Original dataset
    is_training : str
      Label to show whether a data point is in training data or not.
        * "T" -> Training data
        * "F" -> Test data
    Return
    ------
      X : ndarray
        Feature matrix
      y : ndarray
        Lpsa values of each data point.
    '''
    X = dataset.loc[dataset.train == is_training].drop('train', axis=1)
    y = X.pop('lpsa').values
    X = X.to_numpy()
    return X, y


class OneStandardErrorRuleModel:
    '''
    Select the least complex model among one standard error of the best.
    Attributes
    ----------
        estimator :
            A regression model to be parametrized.
        params : dict
            * Keys : The parameter of the model to be chosen by cross-validation.
            * Values : The values for the parameter to be tried.
        cv : Int
            The number of folds for cross-validation.
    '''

    def __init__(self, estimator, params, cv=10):
        self.estimator = estimator
        self.cv = cv
        self.params = params
        self.random_state = 69438  # Seed of the pseudo random number generator

    def fit(self, X, y):
        grid_search = GridSearchCV(self.estimator, self.params,
                                   cv=KFold(self.cv, shuffle=True, random_state=self.random_state),
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search = grid_search.fit(X, y)
        # Gets best estimator according to one standard error rule model
        model_idx = self._get_best_estimator(grid_search.cv_results_)
        self.refit(X, y, model_idx)
        return self

    def _get_best_estimator(self, cv_results):
        cv_mean_errors = -cv_results['mean_test_score']  # Mean errors
        cv_errors = -np.vstack([cv_results[f'split{i}_test_score'] for i in range(self.cv)]).T
        cv_mean_errors_std = np.std(cv_errors, ddof=1, axis=1) / np.sqrt(self.cv)  # Standard errors

        # Finds smallest mean and standard error
        cv_min_error, cv_min_error_std = self._get_cv_min_error(cv_mean_errors, cv_mean_errors_std)

        error_threshold = cv_min_error + cv_min_error_std
        # Finds the least complex model within one standard error of the best
        model_idx = np.argmax(cv_mean_errors < error_threshold)
        cv_mean_error_ = cv_mean_errors[model_idx]
        cv_mean_errors_std_ = cv_mean_errors_std[model_idx]
        return model_idx

    def _get_cv_min_error(self, cv_mean_errors, cv_mean_errors_std):
        # Gets the index of the model with minimum mean error
        best_model_idx = np.argmin(cv_mean_errors)
        cv_min_error = cv_mean_errors[best_model_idx]
        cv_min_error_std = cv_mean_errors_std[best_model_idx]
        return cv_min_error, cv_min_error_std

    def refit(self, X, y, model_idx):
        if self.params:
            param_name = list(self.params.keys())[0]
            self.estimator.set_params(**{param_name: self.params[param_name][model_idx]})
        # Fits the selected model
        self.estimator.fit(X, y)

    def get_test_scores(self, y_test, y_pred):
        y_test, y_pred = y_test.reshape((1, -1)), y_pred.reshape((1, -1))
        errors = (y_test - y_pred) ** 2  # Least sqaure errors
        error = np.mean(errors)  # Mean least sqaure errors
        error_std = np.std(errors, ddof=1) / np.sqrt(y_test.size)  # Standard errors
        return error, error_std


class BestSubsetRegression(LinearRegression):
    '''
    Linear regression based on the best features subset of fixed size.
    Attributes
    ----------
        subset_size : Int
            The number of features in the subset.
    '''

    def __init__(self, subset_size=1):
        LinearRegression.__init__(self)
        self.subset_size = subset_size

    def fit(self, X, y):
        best_combination, best_mse = None, np.inf
        best_intercept_, best_coef_ = None, None
        # Tries all combinations of subset_size
        for combination in combinations(range(X.shape[1]), self.subset_size):
            X_subset = X[:, combination]
            LinearRegression.fit(self, X_subset, y)
            mse = mean_squared_error(y, self.predict(X_subset))
            # Updates the best combination if it gives better result than the current best
            if best_mse > mse:
                best_combination, best_mse = combination, mse
                best_intercept_, best_coef_ = self.intercept_, self.coef_
        LinearRegression.fit(self, X, y)
        # Sets intercept and parameters
        self.intercept_ = best_intercept_
        self.coef_[:] = 0
        self.coef_[list(best_combination)] = best_coef_
        return self


path = 'https://raw.githubusercontent.com/probml/probml-data/main/data/prostate/prostate.csv'
X = pd.read_csv(path, sep='\t').iloc[:, 1:]
X_train, y_train = get_features_and_label(X, 'T')
X_test, y_test = get_features_and_label(X, 'F')

# Standardizes training and test data
scaler = StandardScaler().fit(X.loc[:, 'lcavol':'pgg45'])
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_models, cv, n_alphas = 4, 10, 30
_, n_features = X_train.shape
alpha_lasso, alpha_ridge = [0.680, 0.380, 0.209, 0.100, 0.044, 0.027, 0.012, 0.001], [436, 165, 82, 44, 27, 12, 4,
                                                                                      1e-05]

linear_regression = OneStandardErrorRuleModel(LinearRegression(), {}).fit(X_train, y_train)
bs_regression = OneStandardErrorRuleModel(BestSubsetRegression(), {'subset_size': list(range(1, 9))}).fit(X_train,
                                                                                                          y_train)
ridge_regression = OneStandardErrorRuleModel(Ridge(), {'alpha': alpha_ridge}).fit(X_train, y_train)
lasso_regression = OneStandardErrorRuleModel(Lasso(), {'alpha': alpha_lasso}).fit(X_train, y_train)

regressions = [linear_regression, bs_regression, ridge_regression, lasso_regression]

residuals = np.zeros((X_test.shape[0], n_models))  # (num of test data) x num of models
table = np.zeros(
    (n_features + 3, n_models))  # (num of features + 1(mean error) + 1(std error) + 1(bias coef)) x num of models

for i in range(n_models):
    table[:, i] = regressions[i].estimator.intercept_  # bias
    table[1:n_features + 1, i] = regressions[i].estimator.coef_
    y_pred = regressions[i].estimator.predict(X_test)
    table[n_features + 1:, i] = np.r_[regressions[i].get_test_scores(y_test, y_pred)]
    residuals[:, i] = np.abs(y_test - y_pred)

xlabels = ['Term', 'LS', 'Best Subset', 'Ridge', 'Lasso']  # column headers
row_labels = np.r_[
    [['Intercept']], X.columns[:-2].to_numpy().reshape(-1, 1), [['Test Error'], ['Std Error']]]  # row headers
row_values = np.c_[row_labels, np.round(table, 3)]

fig = plt.figure(figsize=(10, 4))
ax = plt.gca()

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=row_values,
                 colLabels=xlabels,
                 loc='center',
                 cellLoc='center')
table.set_fontsize(20)
table.scale(1.5, 1.5)
fig.tight_layout()
pml.savefig('prostate-subsets-coef.pdf')
plt.show()

plt.figure()
plt.boxplot(residuals)
plt.xticks(np.arange(n_models) + 1, xlabels[1:])
pml.savefig('prostate-subsets-CV.pdf')
plt.show()


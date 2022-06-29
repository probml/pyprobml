#https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/examples/Spam.ipynb

import superimport

import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV, KFold
from typing import Union, List

class OneStandardErrorRuleModel:
    """Select the least complex model within one standard error of the best.
    Parameters
    ----------
        estimator :
            A regression or a classification model to be parametrized.
        param_name :
            The parameter of the model to be chosen by cross-validation.
        param_values :
            The values for the parameter in order of increasing complexity.
        force_model_idx :
            The preselected model index. Sometimes it is useful to run the
            process but to select the specified in advance model.
        n_folds :
            The number of folds for cross-validation.
        is_regression :
            True for a regression model, False for a classification model.
        random_state :
            The seed of the pseudo random number generator.
    Attributes
    ----------
        cv_mean_errors_ :
            Mean CV error for each parameter value.
        cv_mean_errors_std_ :
            Standard error of mean CV error for each parameter value.
        best_model_idx_ :
            The index of the param_values on which minimal mean error achieved.
        cv_min_error_ :
            Mean CV error for the best param_values.
        cv_min_error_std_ :
            Standard error of mean CV error for the best param_values.
        model_idx_ :
            The index of the model selected by "one standard error rule".
        model_ :
            The regression model retrained with selected parameter.
    """
    def __init__(self, estimator, param_name: str, param_values: List[float],
                 force_model_idx: int=None, n_folds: int=10,
                 is_regression: bool=True, random_state: int=69438):
        self.n_folds = n_folds
        self.estimator = estimator
        self.param_name = param_name
        self.param_values = param_values
        self.force_model_idx = force_model_idx
        self.is_regression = is_regression
        # neg_mean_squared_error for regression and accuracy for classification
        self.grid_search = GridSearchCV(
            estimator, {param_name: param_values},
            cv=KFold(n_folds, True, random_state),
            scoring='neg_mean_squared_error' if is_regression else 'accuracy',
            return_train_score=True, iid=True)

    def fit(self, X: np.ndarray, y: np.array) -> 'OneStandardErrorRuleModel':
        self.grid_search.fit(X, y)
        # convert score to mean squared error for regression and to error rate
        # for classification
        cv_errors = -np.vstack(
            [self.grid_search.cv_results_[f'split{i}_test_score']
             for i in range(self.n_folds)]).T
        if not self.is_regression:
            cv_errors = 1 + cv_errors
        # calculate mean error for parameters and their standard error
        self.cv_mean_errors_ = np.mean(cv_errors, axis=1)
        self.cv_mean_errors_std_ = \
            np.std(cv_errors, ddof=1, axis=1) / np.sqrt(self.n_folds)

        # find the best model
        self.best_model_idx_ = np.argmin(self.cv_mean_errors_)
        self.cv_min_error_ = self.cv_mean_errors_[self.best_model_idx_]
        self.cv_min_error_std_ = self.cv_mean_errors_std_[self.best_model_idx_]
        # find the least complex model within one standard error of the best
        error_threshold = self.cv_min_error_ + self.cv_min_error_std_
        self.model_idx_ = np.argmax(self.cv_mean_errors_ < error_threshold)
        return self.__fit_to_all_data(X, y)

    def refit(self, X: np.ndarray, y: np.array,
              force_model_idx: int=None) -> 'OneStandardErrorRuleModel':
        self.force_model_idx = force_model_idx
        return self.__fit_to_all_data(X, y)

    def __fit_to_all_data(self, X, y):
        if self.force_model_idx is not None:
            self.model_idx_ = self.force_model_idx
        self.model_ = self.estimator
        self.model_.set_params(
            **{self.param_name: self.param_values[self.model_idx_]})
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.squeeze(self.model_.predict(X))

    def assess(self, X: np.ndarray, y: np.array) -> (float, float):
        """Calculate mean error of the model and its standard error on the
           specified data."""
        y_hat = self.predict(X)
        errors = (y-y_hat)**2 if self.is_regression else 1.0*(y != y_hat)
        error = np.mean(errors)
        error_std = np.std(errors, ddof=1) / np.sqrt(y.size)
        return error, error_std
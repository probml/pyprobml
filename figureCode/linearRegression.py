#!/usr/bin/env python

from utils import *
import numpy as np


def default_fit_options(reg_type, D):
    """returns an object with default linear regression fit options
    """
    opts = {}
    opts['Display'] = None
    opts['verbose'] = False
    opts['TolFun'] = 1e-3
    opts['MaxIter'] = 200
    opts['Method'] = 'lbfgs'
    opts['MaxFunEvals'] = 2000
    opts['TolX'] = 1e-3
    if reg_type.lower() == 'l1':
        opts['order'] = -1
        if D > 1000:
            opts['corrections'] = 10
    return opts


def preprocessor_apply_to_train(preproc, X):
    """
    Apply Preprocessor to training data and memorize parameters

    preproc is initially a struct with the following fields [default]

    standardize_X - if True, makes columns of X zero mean and unit var. [True]
    rescale_X - if True, scale columns of X to lie in [-1, +1] [False]
    kernel_fn - if not None, apply kernel fn to X  default [None]

    The returned preproc object has several more fields added to it,
    which are used by  preprocessor_apply_to_test
    """

    # Set defaults
    try:
        preproc.standardize_X
    except AttributeError:
        preproc.standardize_X = True
    try:
        preproc.rescale_X
    except AttributeError:
        preproc.rescale_X = False
    try:
        preproc.kernel_fn
    except AttributeError:
        preproc.kernel_fn = None
    try:
        preproc.poly
    except AttributeError:
        preproc.poly = None
    try:
        preproc.add_ones
    except AttributeError:
        preproc.add_ones = None

    if preproc.standardize_X:
        X, preproc.Xmu = center_cols(X)
        X, preproc.Xstnd = mk_unit_variance(X)

    if preproc.rescale_X:
        try:
            preproc.Xscale
        except AttributeError:
            preproc.Xscale = [-1, 1]
        X = rescale_data(X, preproc.Xscale[0], preproc.Xscale[1])

    if preproc.kernel_fn is not None:
        preproc.basis = X
        X = preproc.kernel_fn(X, preproc.basis)

    if preproc.poly is not None:
        assert preproc.poly > 0, 'polynomial degree must be greater than 0'
        X = degexpand(X, preproc.poly, False)

    if preproc.add_ones:
        X = add_ones(X)

    return preproc, X


def preprocessor_apply_to_test(preproc, X):
    """Transform the test data in the same way as the training data"""

    try:
        X = center_cols(X, preproc.Xmu)
    except AttributeError:
        pass

    try:
        X = mk_unit_variance(X, preproc.Xstnd)
    except AttributeError:
        pass

    try:
        X = rescale_data(X, preproc.Xscale[0], preproc.Xscale[1])
    except AttributeError:
        pass

    try:
        if preproc.kernel_fn is not None:
            X = preproc.kernel_fn(X, preproc.basis)
    except AttributeError:
        pass

    try:
        if preproc.poly is not None:
            X = degexpand(X, preproc.poly, False)
    except AttributeError:
        pass

    try:
        if preproc.add_ones:
            X = add_ones(X)
    except AttributeError:
        pass

    return X


def linreg_create():
    pass


def linreg_fit(X, y, **kwargs):
    """
    Fit a linear regression model with MLE or MAP.
    This is a port of linregFit.m from pmtk3.

    :param X: N*D design matrix
    :param y: N*1 response vector
    """
    pp = preprocessor_create(add_ones=True, standardize_X=False)  # default

    N = len(X)
    D = 1 if len(X.shape) < 2 else X.shape[1]

    weights = kwargs['weights'] if 'weights' in kwargs else np.ones(N)
    reg_type = kwargs['reg_type'] if 'reg_type' in kwargs else None
    likelihood = kwargs['likelihood'] if 'likelihood' in kwargs else 'gaussian'
    lambda_ = kwargs['lambda_'] if 'lambda_' in kwargs else None
    fit_options = kwargs['fit_options'] if 'fit_options' in kwargs else None
    preproc = kwargs['preproc'] if 'preproc' in kwargs else pp
    fit_fn_name = kwargs['fit_fn_name'] if 'fit_fn_name' in kwargs else None
    winit = kwargs['winit'] if 'winit' in kwargs else None

    if preproc is None:
        preproc = preprocessor_create()

    if preproc.add_ones:
        D += 1

    if winit is None:
        winit = np.zeros(D)

    if reg_type is None:
        if lambda_ is None:
            # MLE
            reg_type = 'l2'
            lambda_ = 0
        else:
            #L2
            reg_type = 'l2'

    if fit_options is None:
        fit_options = default_fit_options(reg_type, D)

    if fit_fn_name is None:
        if reg_type.lower() == 'l1':
            fit_fn_name = 'l1GeneralProjection'
        elif reg_type.lower() == 'l2':
            fit_fn_name = 'qr'

    model = {}

    if likelihood.lower() == 'huber':
        raise NotImplementedError
    elif likelihood.lower() == 'student':
        raise NotImplementedError
    elif likelihood.lower() == 'gaussian':
        preproc, X = preprocessor_apply_to_train(preproc, X)
        N = len(X)
        D = 1 if len(X.shape) < 2 else X.shape[1]
        model['lambda_'] = lambda_
        lambda_vec = lambda_ * np.ones(D)

        if preproc.add_ones:
            lambda_vec[0] = 0  # don't penalize bias term

        winit = np.zeros(D)
        opts = fit_options

        if reg_type == 'l1':
            raise NotImplementedError
        elif reg_type == 'l2':
            if fit_fn_name == 'qr':
                if lambda_ == 0:
                    R = np.diag(np.sqrt(weights))
                    RX = R.dot(X)
                    w = np.linalg.pinv(RX.T.dot(RX)).dot(RX.T).dot(R.dot(y))
                else:
                    raise NotImplementedError
            elif fit_fn_name == 'minfunc':
                raise NotImplementedError
            else:
                raise ValueError('Invalid fit function')
        elif reg_type == 'scad':
            raise NotImplementedError
        else:
            raise ValueError('Invalid regression type')
    else:
        raise ValueError('Invalid likelihood')

    model['w'] = w
    yhat = X.dot(w)

    if weights.sum() == 0:
        model['sigma2'] = np.spacing(1)
    else:
        model['sigma2'] = np.sum(weights * np.square(y - yhat)) / \
            np.sum(weights)

    model['preproc'] = preproc
    model['model_type'] = 'linreg'
    model['likelihood'] = likelihood

    return model


def linreg_fit_bayes(X, y, **kwargs):
    """
    Fit a Bayesian linear regression model.
    This is a port of linregFit.m from pmtk3.

    :param X: N*D design matrix
    :param y: N*1 response vector
    """
    pp = preprocessor_create(add_ones=True, standardize_X=False)  # default

    prior = kwargs['prior'] if 'prior' in kwargs else 'uninf'
    preproc = kwargs['preproc'] if 'preproc' in kwargs else pp
    beta = kwargs['beta'] if 'beta' in kwargs else None
    alpha = kwargs['alpha'] if 'alpha' in kwargs else None
    g = kwargs['g'] if 'g' in kwargs else None
    use_ARD = kwargs['use_ARD'] if 'use_ARD' in kwargs else False
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    if prior.lower() == 'eb':
        prior = 'ebnetlab'

    if prior.lower() == 'uninf':
        raise NotImplementedError
    elif prior.lower() == 'gauss':
        raise NotImplementedError
    elif prior.lower() == 'zellner':
        raise NotImplementedError
    elif prior.lower() == 'vb':
        raise NotImplementedError
    elif prior.lower() == 'ebnetlab':
        model, logev = linreg_fit_eb_netlab(X, y, preproc)
    elif prior.lower() == 'ebchen':
        raise NotImplementedError
    else:
        raise ValueError('Invalid prior')

    model['model_type'] = 'linreg_bayes'
    model['prior'] = prior

    return model, logev


def linreg_fit_path_cv():
    pass


def linreg_logprob():
    pass


def linreg_predict(model, X, v=False):
    """
    Prediction with linear regression
    yhat[i] = E[y|X[i, :]], model]
    v[i] = Var[y|X[i, :], model]
    """
    if 'preproc' in model:
        X = preprocessor_apply_to_test(model['preproc'], X)

    yhat = X.dot(model['w'])
    return yhat


def linreg_predict_bayes():
    pass

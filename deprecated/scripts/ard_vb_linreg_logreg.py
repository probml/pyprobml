# Bayesian Linear and logistic Regression with ARD prior (fitted with Variational Bayes)

#https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/vrvm.py

import superimport

import numpy as np
from sklearn.externals import six
from scipy.special import expit
from scipy.linalg import solve_triangular
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils import check_X_y,check_array
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import as_float_array
import warnings


class VBRegressionARD(LinearModel,RegressorMixin):
    '''
    Bayesian Linear Regression with ARD prior (fitted with Variational Bayes)
    
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations
    fit_intercept : boolean, optional (DEFAULT = True)
        If True, intercept will be used in computation
        
    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.
        
    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied, otherwise it will be overwritten.
        
    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model 
       
    a: float, optional, (DEFAULT = 1e-5)
       Shape parameters for Gamma distributed precision of weights
       
    b: float, optional, (DEFAULT = 1e-5)
       Rate parameter for Gamma distributed precision of weights
    
    c: float, optional, (DEFAULT = 1e-5)
       Shape parameter for Gamma distributed precision of noise
    
    d: float, optional, (DEFAULT = 1e-5)
       Rate parameter for Gamma distributed precision of noise
       
    prune_thresh: float, ( DEFAULT = 1e-3 )
       Threshold for pruning out variable (applied after model is fitted)
    
    
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
    active_ : array, dtype = np.bool, shape = (n_features)
        True for non-zero coefficients, False otherwise
    sigma_ : array, shape = (n_features, n_features)
        Estimated covariance matrix of the weights, computed only
         for non-zero coefficients
       
    Reference:
    ----------
    [1] Bishop & Tipping (2000), Variational Relevance Vector Machine
    [2] Jan Drugowitch (2014), Variational Bayesian Inference for Bayesian Linear 
                           and Logistic Regression
    [3] Bishop (2006) Pattern Recognition and Machine Learning (ch. 7)
    '''
    def __init__(self,  n_iter = 100, tol = 1e-3, fit_intercept = True,
                 a = 1e-5, b = 1e-5, c = 1e-5, d = 1e-5, copy_X = True, 
                 prune_thresh = 1e-3, verbose = False):
        self.n_iter          = n_iter
        self.tol             = tol
        self.fit_intercept   = fit_intercept
        self.a,self.b        = a,b
        self.c,self.d        = c,d
        self.copy_X          = copy_X
        self.verbose         = verbose
        self.prune_thresh    = prune_thresh
        
        
    def _center_data(self,X,y):
        ''' Centers data'''
        X     = as_float_array(X,self.copy_X)
        # normalisation should be done in preprocessing!
        X_std = np.ones(X.shape[1], dtype = X.dtype)
        if self.fit_intercept:
            X_mean = np.average(X,axis = 0)
            y_mean = np.average(y,axis = 0)
            X     -= X_mean
            y      = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1],dtype = X.dtype)
            y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
        return X,y, X_mean, y_mean, X_std
        
        
    def fit(self,X,y):
        '''
        Fits variational relevance ARD regression
                
        Parameters
        -----------
        X: array-like of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples, n_features] 
           Target values
           
        Returns
        -------
        self : object
            Returns self.
        '''
        # precompute some values for faster iterations 
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape
        X, y, X_mean, y_mean, X_std = self._center_data(X, y)
        XX               = np.dot(X.T,X)
        XY               = np.dot(X.T,y)
        Y2               = np.sum(y**2)
        
        # final update for a and c
        a        = (self.a + 0.5) * np.ones(n_features, dtype = np.float)
        c        = (self.c + 0.5 * n_samples) #* np.ones(n_features, dtype = np.float)
        # initial values of b,d before mean field approximation
        d        = self.d #* np.ones(n_features, dtype = np.float)
        b        = self.b * np.ones(n_features, dtype = np.float)
        active   = np.ones(n_features, dtype = np.bool)
        w0       = np.zeros(n_features) 
        w        = np.copy(w0)
        
        for i in range(self.n_iter):
            # ----------------------  update q(w) -----------------------
            
            # calculate expectations for precision of noise & precision of weights
            e_tau   = c / d
            e_A     = a / b
            XXa     = XX[active,:][:,active]
            XYa     = XY[active]
            Xa      = X[:,active]
            # parameters of updated posterior distribution
            w[active],Ri  = self._posterior_weights(XXa,XYa,e_tau,e_A[active])
                
            # --------------------- update q(tau) ------------------------
            # update rate parameter for Gamma distributed precision of noise 
            XSX       = np.sum( np.dot(Xa,Ri.T)**2)
            XMw       = np.sum( np.dot(Xa,w[active])**2 )    
            XYw       = np.sum( w[active]*XYa )
            d         = self.d + 0.5*(Y2 + XMw + XSX) - XYw
            
            # -------------------- update q(alpha(j)) for each j ----------
            # update rate parameter for Gamma distributed precision of weights
            b[active] = self.b + 0.5*(w[active]**2 + np.sum(Ri**2,axis = 1))
            
            # -------------------- check convergence ----------------------
            # determine relevant vector as is described in Bishop & Tipping 
            # (i.e. using mean of posterior distribution)
            active  = np.abs(w) > self.prune_thresh
            
            # make sure there is at least one relevant feature
            if np.sum(active) == 0:
                active[np.argmax(np.abs(w))] = True
            # all irrelevant features are forced to zero
            w[~active] = 0
            # check convergence
            if np.sum(abs(w-w0) > self.tol) == 0 or i==self.n_iter-1:
                break
            w0 = np.copy(w)
            # if only one relevant feature => terminate
            if np.sum(active)== 1:
                if X.shape[1] > 3 and self.prune_thresh > 1e-1:
                   warnings.warn(("Only one relevant feature found! it can be useful to decrease"
                                  "value for parameter prune_thresh"))
                break
            
        # update parameters after last update
        e_tau         = c / d
        e_A           = a / b 
        XXa           = XX[active,:][:,active]
        XYa           = XY[active]
        w[active], self.sigma_ = self._posterior_weights(XXa,XYa,e_tau,e_A[active],True)
        self._e_tau_  = e_tau        
        self.coef_    = w
        self._set_intercept(X_mean,y_mean,X_std)
        self.active_  = active 
        return self
        
        
        
    def predict_dist(self,X):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and standard
        deviation.
        
        Parameters
        -----------
        X: {array-like, sparse} [n_samples_test, n_features]
           Test data, matrix of explanatory variables
           
        Returns
        -------
        y_hat: numpy array of size (n_samples_test,)
           Estimated values of targets on test set (Mean of predictive distribution)
           
        var_hat: numpy array of size (n_samples_test,)
           Error bounds (Standard deviation of predictive distribution)
        '''
        y_hat        = self._decision_function(X)
        data_noise   = 1./self._e_tau_
        model_noise  = np.sum( np.dot(X[:,self.active_],self.sigma_) * X[:,self.active_], axis = 1)
        var_hat      = data_noise + model_noise        
        return y_hat, var_hat
        
    
    
    def _posterior_weights(self, XX, XY, exp_tau, exp_A, full_covar = False):
        '''
        Calculates parameters of posterior distribution of weights
        
        Parameters:
        -----------
        X:  numpy array of size n_features
            Matrix of active features (changes at each iteration)
        
        XY: numpy array of size [n_features]
            Dot product of X and Y (for faster computations)
        exp_tau: float
            Mean of precision parameter of noise
            
        exp_A: numpy array of size n_features
            Vector of precisions for weights
           
        Returns:
        --------
        [Mw, Sigma]: list of two numpy arrays
        
        Mw: mean of posterior distribution
        Sigma: covariance matrix
        '''
        # compute precision parameter
        S    = exp_tau*XX       
        np.fill_diagonal(S, np.diag(S) + exp_A)
        
        # cholesky decomposition
        R    = np.linalg.cholesky(S)
        
        # find mean of posterior distribution
        RtMw = solve_triangular(R, exp_tau*XY, lower = True, check_finite = False)
        Mw   = solve_triangular(R.T, RtMw, lower = False, check_finite = False)
        
        # use cholesky decomposition of S to find inverse ( or diagonal of inverse)
        Ri    = solve_triangular(R, np.eye(R.shape[1]), lower = True, check_finite = False)
        if full_covar:
            Sigma = np.dot(Ri.T,Ri)
            return [Mw,Sigma]
        else:
            return [Mw,Ri]


#----------------------   Classification   ---------------------------------------------



def lam(eps):
    ''' 
    Calculates lambda eps [part of local variational approximation
    to sigmoid function]
    '''
    return 0.5 / eps * ( expit(eps) - 0.5 )
    

class VBClassificationARD(LinearClassifierMixin, BaseEstimator):
    '''
    Variational Bayesian Logistic Regression with local variational approximation.
    
    
    Parameters:
    -----------
    n_iter: int, optional (DEFAULT = 50 )
       Maximum number of iterations
       
    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold, if cange in coefficients is less than threshold
       algorithm is terminated
    
    fit_intercept: bool, optinal ( DEFAULT = True )
       If True uses bias term in model fitting
       
    a: float, optional (DEFAULT = 1e-6)
       Rate parameter for Gamma prior on precision parameter of coefficients
       
    b: float, optional (DEFAULT = 1e-6)
       Shape parameter for Gamma prior on precision parameter of coefficients
       
    prune_thresh: float, optional (DEFAULT = 1e-4)
       Threshold for pruning out variable (applied after model is fitted)
    
    verbose: bool, optional (DEFAULT = False)
       Verbose mode
       
       
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients
    
    intercept_: array, shape = (n_features)
        intercepts
        
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise        
    References:
    -----------
    [1] Bishop 2006, Pattern Recognition and Machine Learning ( Chapter 7,10 )
    [2] Murphy 2012, Machine Learning A Probabilistic Perspective ( Chapter 14,21 )
    [3] Bishop & Tipping 2000, Variational Relevance Vector Machine
    '''
    def __init__(self,  n_iter = 100, tol = 1e-3, fit_intercept = True,
                 a = 1e-4, b = 1e-4, prune_thresh = 1e-4, verbose = True):
        self.n_iter            = n_iter
        self.tol               = tol
        self.verbose           = verbose
        self.prune_thresh      = prune_thresh
        self.fit_intercept     = fit_intercept
        self.a                 = a
        self.b                 = b
        
        
    def fit(self,X,y):
        '''
        Fits variational Bayesian Logistic Regression with local variational bound
        
        Parameters
        ----------
        X: array-like of size (n_samples, n_features)
           Matrix of explanatory variables
           
        y: array-like of size (n_samples,)
           Vector of dependent variables
        Returns
        -------
        self: object
           self
        '''
        # preprocess data
        X,y = check_X_y( X, y , dtype = np.float64)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # take into account bias term if required 
        n_samples, n_features = X.shape
        n_features = n_features + int(self.fit_intercept)
        if self.fit_intercept:
            X = np.hstack( (np.ones([n_samples,1]),X))
        
        # handle multiclass problems using One-vs-Rest 
        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes")
        if n_classes > 2:
            self.coef_, self.sigma_ = [0]*n_classes,[0]*n_classes
            self.intercept_         = [0]*n_classes
            self.active_            = [0]*n_classes
        else:
            self.coef_, self.sigma_, self.intercept_ = [0],[0],[0]
            self.active_ = [0]
        
        # hyperparameters of precision for weights
        a  = self.a + 0.5 * np.ones(n_features)
        b  = self.b * np.ones(n_features)
        
        for i in range(len(self.coef_)):
            if n_classes == 2:
                pos_class = self.classes_[1]
            else:
                pos_class   = self.classes_[i]
            mask            = (y == pos_class)
            y_bin           = np.ones(y.shape)
            y_bin[~mask]    = 0
            coef_, sigma_, intercept_, active_   = self._fit(X,y_bin,a,b)
            self.coef_[i]      = coef_
            self.intercept_[i] = intercept_
            self.sigma_[i]     = sigma_
            self.active_[i]    = active_
            
        self.coef_  = np.asarray(self.coef_)
        return self
        

    def predict_proba(self,x):
        '''
        Predicts probabilities of targets for test set
        
        Parameters
        ----------
        X: array-like of size [n_samples_test,n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        -------
        probs: numpy array of size [n_samples_test]
           Estimated probabilities of target classes
        '''
        scores = self.decision_function(x)
        if self.fit_intercept:
            x = np.hstack( (np.ones([x.shape[0],1]),x))
        var = [np.sum(np.dot(x[:,a],s)*x[:,a],axis = 1) for a,s in zip(self.active_,self.sigma_)]
        sigma  = np.asarray(var)
        ks = 1. / ( 1. + np.pi*sigma / 8)**0.5
        probs = expit(scores.T*ks).T
        if probs.shape[1] == 1:
            probs =  np.hstack([1 - probs, probs])
        else:
            probs /= np.reshape(np.sum(probs, axis = 1), (probs.shape[0],1))
        return probs

            
    def _fit(self,X,y,a,b):
        '''
        Fits single classifier for each class (for OVR framework)
        '''
        eps     = 1 # default starting parameter for Jaakola Jordan bound
        w0      = np.zeros(X.shape[1])
        w       = np.copy(w0)
        active  = np.ones(X.shape[1], dtype = np.bool)
        XY      = np.dot(X.T, y - 0.5)
        
        for i in range(self.n_iter):
            # In the E-step we update approximation of 
            # posterior distribution q(w,alpha) = q(w)*q(alpha)

            # --------- update q(w) ------------------
            l       = lam(eps)
            Xa      = X[:,active]
            XYa     = XY[active]   #np.dot(Xa.T,(y-0.5))
            w[active],Ri = u,v   = self._posterior_dist(Xa,l,a[active],b[active],XYa)
        
            # -------- update q(alpha) ---------------
            b[active] = self.b + 0.5*(w[active]**2 + np.sum(Ri**2,1))
            
            # -------- update eps  ------------
            # In the M-step we update parameter eps which controls 
            # accuracy of local variational approximation to lower bound
            XMX = np.dot(Xa,w[active])**2
            XSX = np.sum( np.dot(Xa,Ri.T)**2, axis = 1)
            eps = np.sqrt( XMX + XSX )
            
            # determine relevant vector as is described in Bishop & Tipping 
            # (i.e. using mean of posterior distribution)
            active  = np.abs(w) > self.prune_thresh
            
            # do not prune intercept & make sure there is at least one 'relevant feature'.
            # If only one relevant feature , then choose rv with largest posterior mean
            if self.fit_intercept:
                active[0] = True
                if np.sum(active[1:]) == 0:
                    active[np.argmax(np.abs(w[1:]))] = True
            else:
                if np.sum(active) == 0:
                    active[np.argmax(np.abs(w))] = True
            # all irrelevant features are forced to zero
            w[~active] = 0
            # check convergence
            if np.sum(abs(w-w0) > self.tol) == 0 or i==self.n_iter-1:
                break
            w0 = np.copy(w)
            # if only one relevant feature => terminate
            if np.sum(active) - 1*self.fit_intercept == 1:
                if X.shape[1] > 3 and self.prune_thresh > 1e-1:
                   warnings.warn(("Only one relevant feature found! it can be useful to decrease"
                                  "value for parameter prune_thresh"))
                break
            
        l   = lam(eps)
        Xa  = X[:,active]
        XYa = np.dot(Xa.T,(y-0.5)) 
        w[active] , sigma_  = self._posterior_dist(Xa,l,a[active],b[active],XYa,True)
        
        # separate intercept & coefficients
        intercept_ = 0
        if self.fit_intercept:
            intercept_ = w[0]
            coef_      = np.copy(w[1:])
        else:
            coef_      = w
        return coef_, sigma_ , intercept_, active


    def _posterior_dist(self,X,l,a,b,XY,full_covar = False):
        '''
        Finds gaussian approximation to posterior of coefficients
        '''
        sigma_inv  = 2*np.dot(X.T*l,X)
        alpha_vec  = a / b
        if self.fit_intercept:
            alpha_vec[0] = np.finfo(np.float64).eps
        np.fill_diagonal(sigma_inv, np.diag(sigma_inv) + alpha_vec)
        R     = np.linalg.cholesky(sigma_inv)
        Z     = solve_triangular(R,XY, lower = True)
        mean_ = solve_triangular(R.T,Z,lower = False)
        Ri    = solve_triangular(R,np.eye(X.shape[1]), lower = True)
        if full_covar:
            sigma_   = np.dot(Ri.T,Ri)
            return mean_ , sigma_
        else:
            return mean_ , Ri



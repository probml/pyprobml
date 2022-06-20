# Variational Bayes for binary logistic regression using JJ bound and mean field Gaussian-Gamma.
# Also, Laplace approx with EB for multiclass.
# Written by Amazasp Shaumyan
#https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/bayes_logistic.py

import superimport

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.optimize import newton_cg
from scipy.special import expit, exprel
from scipy.linalg import eigvalsh
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from scipy.linalg import solve_triangular
from sklearn.linear_model.logistic import ( _logistic_loss_and_grad, _logistic_loss, 
                                            _logistic_grad_hess,)



class BayesianLogisticRegression(LinearClassifierMixin, BaseEstimator):
    '''
    Superclass for two different implementations of Bayesian Logistic Regression
    ''' 
    def __init__(self, n_iter, tol, fit_intercept, verbose):
        self.n_iter        = n_iter
        self.tol           = tol
        self.fit_intercept = fit_intercept
        self.verbose       = verbose
    
    
    def fit(self,X,y):
        '''
        Fits Bayesian Logistic Regression
        Parameters
        -----------
        X: array-like of size (n_samples, n_features)
           Training data, matrix of explanatory variables
        
        y: array-like of size (n_samples, )
           Target values
           
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
        
        # prepare for ovr if required        
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = self._add_intercept(X)
            
        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes")
        if n_classes > 2:
            self.coef_, self.sigma_ = [0]*n_classes,[0]*n_classes
            self.intercept_         = [0]*n_classes
        else:
            self.coef_, self.sigma_, self.intercept_ = [0],[0],[0]

        # make classifier for each class (one-vs-the rest)
        for i in range(len(self.coef_)):
            if n_classes == 2:
                pos_class = self.classes_[1]
            else:
                pos_class = self.classes_[i]
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.float64)
            y_bin[~mask] = self._mask_val
            coef_, sigma_ = self._fit(X,y_bin)
            if self.fit_intercept:
                self.intercept_[i],self.coef_[i] = self._get_intercept(coef_)
            else:
                self.coef_[i]      = coef_
            self.sigma_[i] = sigma_
            
        self.coef_  = np.asarray(self.coef_)
        return self
        
        
    def predict_proba(self,X):
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
        # construct separating hyperplane
        scores = self.decision_function(X)
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # probit approximation to predictive distribution
        sigma = self._get_sigma(X)
        ks = 1. / ( 1. + np.pi*sigma / 8)**0.5
        probs = expit(scores.T*ks).T

        # handle several class cases
        if probs.shape[1] == 1:
            probs =  np.hstack([1 - probs, probs])
        else:
            probs /= np.reshape(np.sum(probs, axis = 1), (probs.shape[0],1))
        return probs
        
        
    def _add_intercept(self,X):
        '''Adds intercept to data matrix'''
        raise NotImplementedError
        
        
    def _get_intercept(self,coef):
        '''
        Extracts value of intercept from coefficients
        '''
        raise NotImplementedError
        
        
    def _get_sigma(self,X):
        '''
        Computes variance of predictive distribution (which is then used in 
        probit approximation of sigmoid)
        '''
        raise NotImplementedError



class EBLogisticRegression(BayesianLogisticRegression):
    '''
    Implements Bayesian Logistic Regression with type II maximum likelihood 
    (sometimes it is called Empirical Bayes), uses Gaussian (Laplace) method 
    for approximation of evidence function.
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 50)
        Maximum number of iterations before termination
        
    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.
        
    solver: str, optional (DEFAULT = 'lbfgs_b')
        Optimization method that is used for finding parameters of posterior
        distribution ['lbfgs_b','newton_cg']
        
    n_iter_solver: int, optional (DEFAULT = 15)
        Maximum number of iterations before termination of solver
        
    tol_solver: float, optional (DEFAULT = 1e-3)
        Convergence threshold for solver (it is used in estimating posterior
        distribution), 
    fit_intercept : bool, optional ( DEFAULT = True )
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations
        
    alpha: float (DEFAULT = 1e-6)
        Initial regularization parameter (precision of prior distribution)
        
    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model
        
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
    sigma_ : array, shape = (n_features, )
        eigenvalues of covariance matrix
        
    alpha_: float
        Precision parameter of weight distribution
    
    intercept_: array, shape = (n_features)
        intercept
    
    References:
    -----------
    [1] Pattern Recognition and Machine Learning, Bishop (2006) (pages 293 - 294)
    '''
    
    def __init__(self, n_iter = 50, tol = 1e-3,solver = 'lbfgs_b',n_iter_solver = 15,
                 tol_solver = 1e-3, fit_intercept = True, alpha = 1e-6,  verbose = False):
        super(EBLogisticRegression,self).__init__(n_iter, tol, fit_intercept, verbose)
        self.n_iter_solver     = n_iter_solver
        self.tol_solver        = tol_solver
        self.alpha             = alpha
        if solver not in ['lbfgs_b','newton_cg']:
            raise ValueError(('Only "lbfgs_b" and "newton_cg" '
                              'solvers are implemented'))
        self.solver            = solver
        # masking value (this is set for use in lbfgs_b and newton_cg)
        self._mask_val         = -1.
        

    def _fit(self,X,y):
        '''
        Maximizes evidence function (type II maximum likelihood) 
        '''
        # iterative evidence maximization
        alpha = self.alpha
        n_samples,n_features = X.shape
        w0 = np.zeros(n_features)
        
        for i in range(self.n_iter):
            
            alpha0 = alpha
            
            # find mean & covariance of Laplace approximation to posterior
            w, d   = self._posterior(X, y, alpha, w0) 
            mu_sq  = np.sum(w**2)
            
            # use Iterative updates for Bayesian Logistic Regression
            # Note in Bayesian Logistic Gull-MacKay fixed point updates 
            # and Expectation - Maximization algorithm give the same update
            # rule
            alpha = X.shape[1] / (mu_sq + np.sum(d))
            
            # check convergence
            delta_alpha = abs(alpha - alpha0)
            if delta_alpha < self.tol or i==self.n_iter-1:
                break
            
        # after convergence we need to find updated MAP vector of parameters
        # and covariance matrix of Laplace approximation
        coef_, sigma_ = self._posterior(X, y, alpha , w)
        self.alpha_ = alpha
        return coef_, sigma_
            

    def _add_intercept(self,X):
        '''
        Adds intercept to data (intercept column is not used in lbfgs_b or newton_cg
        it is used only in Hessian)
        '''
        return np.hstack((X,np.ones([X.shape[0],1])))
        
    
    def _get_intercept(self,coef):
        '''
        Returns intercept and coefficients
        '''
        return coef[-1], coef[:-1]
        
        
    def _get_sigma(self,X):
        ''' Compute variance of predictive distribution'''
        return np.asarray([ np.sum(X**2*s,axis = 1) for s in self.sigma_])
    
            
    def _posterior(self, X, Y, alpha0, w0):
        '''
        Iteratively refitted least squares method using l_bfgs_b or newton_cg.
        Finds MAP estimates for weights and Hessian at convergence point
        '''
        n_samples,n_features  = X.shape
        if self.solver == 'lbfgs_b':
            f = lambda w: _logistic_loss_and_grad(w,X[:,:-1],Y,alpha0)
            w = fmin_l_bfgs_b(f, x0 = w0, pgtol = self.tol_solver,
                              maxiter = self.n_iter_solver)[0]
        elif self.solver == 'newton_cg':
            f    = _logistic_loss
            grad = lambda w,*args: _logistic_loss_and_grad(w,*args)[1]
            hess = _logistic_grad_hess               
            args = (X[:,:-1],Y,alpha0)
            w    = newton_cg(hess, f, grad, w0, args=args,
                             maxiter=self.n_iter, tol=self.tol)[0]
        else:
            raise NotImplementedError('Liblinear solver is not yet implemented')
            
        # calculate negative of Hessian at w
        xw    = np.dot(X,w)
        s     = expit(xw)
        R     = s * (1 - s)
        Hess  = np.dot(X.T*R,X)    
        Alpha = np.ones(n_features)*alpha0
        if self.fit_intercept:
            Alpha[-1] = np.finfo(np.float16).eps
        np.fill_diagonal(Hess, np.diag(Hess) + Alpha)
        e  =  eigvalsh(Hess)        
        return w,1./e


#============== VB Logistic Regression (with Jaakola Jordan bound) ==================



def lam(eps):
    ''' Calculates lambda eps (used for Jaakola & Jordan local bound) '''
    eps = -abs(eps)
    return 0.25 * exprel(eps) / (np.exp(eps) + 1)
    


class VBLogisticRegression(BayesianLogisticRegression):
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
        
    References:
    -----------
   [1] Bishop 2006, Pattern Recognition and Machine Learning ( Chapter 10 )
   [2] Murphy 2012, Machine Learning A Probabilistic Perspective ( Chapter 21 )
    '''
    def __init__(self,  n_iter = 50, tol = 1e-3, fit_intercept = True,
                 a = 1e-4, b = 1e-4, verbose = True):
        super(VBLogisticRegression,self).__init__(n_iter, tol, fit_intercept, verbose)
        self.a                 =  a
        self.b                 =  b
        self._mask_val         =  0.
       
            
    def _fit(self,X,y):
        '''
        Fits single classifier for each class (for OVR framework)
        '''
        eps = 1
        n_samples, n_features = X.shape
        XY  = np.dot( X.T , (y-0.5))
        w0  = np.zeros(n_features)
        
        # hyperparameters of q(alpha) (approximate distribution of precision
        # parameter of weights)
        a  = self.a + 0.5 * n_features
        b  = self.b
  
        for i in range(self.n_iter):
            # In the E-step we update approximation of 
            # posterior distribution q(w,alpha) = q(w)*q(alpha)
            
            # --------- update q(w) ------------------
            l  = lam(eps)
            w,Ri = self._posterior_dist(X,l,a,b,XY)
                        
            # -------- update q(alpha) ---------------
            if self.fit_intercept:
                b = self.b + 0.5*(np.sum(w[1:]**2) + np.sum(Ri[1:,:]**2))
            else:
                b = self.b + 0.5*(np.sum(w**2) + np.sum(Ri**2))
            
            # -------- update eps  ------------
            # In the M-step we update parameter eps which controls 
            # accuracy of local variational approximation to lower bound
            XMX = np.dot(X,w)**2
            XSX = np.sum( np.dot(X,Ri.T)**2, axis = 1)
            eps = np.sqrt( XMX + XSX )
            
            #print('iter {}'.format(i))
            #print(a)
            #print(b)
            #print(w)
            
            # convergence
            if np.sum(abs(w-w0) > self.tol) == 0 or i==self.n_iter-1:
                break
            w0 = w
            
        l  = lam(eps)
        coef_, sigma_  = self._posterior_dist(X,l,a,b,XY,True)
        self.a = a
        self.b = b
        return coef_, sigma_


    def _add_intercept(self,X):
        '''Adds intercept to data matrix'''
        return np.hstack((np.ones([X.shape[0],1]),X))
        
    
    def _get_intercept(self, coef):
        ''' Returns intercept and coefficients '''
        return coef[0], coef[1:]
        
        
    def _get_sigma(self,X):
        ''' Compute variance of predictive distribution'''
        return np.asarray([np.sum(np.dot(X,s)*X,axis = 1) for s in self.sigma_])


    def _posterior_dist(self,X,l,a,b,XY,full_covar = False):
        '''
        Finds gaussian approximation to posterior of coefficients using
        local variational approximation of Jaakola & Jordan
        '''
        sigma_inv  = 2*np.dot(X.T*l,X)
        alpha_vec  = np.ones(X.shape[1])*float(a) / b
        if self.fit_intercept:
            alpha_vec[0] = np.finfo(np.float16).eps
        np.fill_diagonal(sigma_inv, np.diag(sigma_inv) + alpha_vec)
        R     = np.linalg.cholesky(sigma_inv)
        Z     = solve_triangular(R,XY, lower = True)
        mean  = solve_triangular(R.T,Z,lower = False)
        
        # is there any specific function in scipy that efficently inverts
        # low triangular matrix ????
        Ri    = solve_triangular(R,np.eye(X.shape[1]), lower = True)
        if full_covar:
            sigma   = np.dot(Ri.T,Ri)
            return mean, sigma
        else:
            return mean , Ri
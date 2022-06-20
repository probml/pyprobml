#  Solving  linear systems


import superimport

import numpy as np
from scipy import linalg
from itertools import combinations



def naive_solve(A, b):
    return np.linalg.inv(A.T @ A) @ A.T @ b

def qr_solve(A, b):
    m, n = np.shape(A)
    if m > n:
        return qr_solve_over(A, b)
    else:
        return qr_solve_under(A, b)
    
def qr_solve_over(A, b):
    Q, R = np.linalg.qr(A) 
    Qb = np.dot(Q.T,b) 
    return scipy.linalg.solve_triangular(R, Qb)

def qr_solve_under(A, b):
    Q, R = np.linalg.qr(A.T) 
    c = scipy.linalg.solve_triangular(R, b)
    m, n = A.shape
    x = np.zeros(n)
    xx = np.dot(Q, c)
    xx = xx[:,0]
    K = xx.shape[0]
    x[:K] = xx # other components are zero
    return x



def run_demo(m, n):
    print('Solving linear system with {} constraints and {} unknowns'.format(m, n))
    np.random.seed(0)
    A = np.random.rand(m,n)
    b = np.random.rand(m,1) 
    
    methods = list()
    solns = list()
    
    methods.append('naive')
    solns.append(naive_solve(A, b))
    
    methods.append('pinv')
    solns.append(np.dot(np.linalg.pinv(A), b))
    
    methods.append('lstsq')
    solns.append(np.linalg.lstsq(A, b, rcond=None)[0])
    
    methods.append('qr')
    solns.append(qr_solve(A, b))
    
    
    for (method, soln) in zip(methods, solns):
        residual = b -  np.dot(A, soln)
        print('method {}, norm {:0.5f}, residual {:0.5f}'.format(method, np.linalg.norm(soln), np.linalg.norm(residual)))
        print(soln.T)
    
    # https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator
    if m < n: # underdetermined
        rank = np.linalg.matrix_rank(A)
        assert m==rank
        for nz in combinations(range(n), rank):    # the variables not set to zero
            soln = np.zeros((n,1))
            soln[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
            residual = b -  np.dot(A, soln)
            print('sparse qr, norm {:0.5f}, residual {:0.5f}'.format(np.linalg.norm(soln), np.linalg.norm(residual)))
            print(soln.T) 
    
m = 5; n = 3 # Overdetermined
run_demo(m, n)

m = 3; n = 5 # Underdetermined
run_demo(m, n)

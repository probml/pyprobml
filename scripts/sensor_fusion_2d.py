import superimport

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import pyprobml_utils as pml

#figdir = "../figures";
#def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

def gauss_plot2d(mu, sigma, plot_options):
    plt.scatter(mu[0], mu[1],marker="x", c=plot_options['color'])
    plt.plot(*cov_to_pts(sigma)+mu.reshape(2,1), '-o',c=plot_options['color'], markersize=0.1)

def cov_to_pts( cov ):
    """helper function to get covariance interval for plotting, this can likely be included in the utils folder"""
    circ = np.linspace( 0, 2*np.pi, 100 )
    sf = np.asarray( [ np.cos( circ ), np.sin( circ ) ] )
    [u,s,v] = np.linalg.svd( cov )
    pmat = u*2.447*np.sqrt(s) # 95% confidence
    return np.dot(  pmat, sf )

def gauss_soft_condition(pmu, py, A, y):
    sy_inv = np.linalg.inv(py['sigma'])
    smu_inv = np.linalg.inv(pmu['sigma'])
    post = {}
    post['sigma'] = np.linalg.inv(smu_inv + A.T.dot(sy_inv).dot(A))

    # reshape is needed to assist in + broadcasting
    ny = py['mu'].shape[0] # 4
    nm = pmu['mu'].shape[0] # 2 
    post['mu'] = post['sigma'].dot(A.T.dot(sy_inv).dot(y.reshape(ny,1) - py['mu']) + 
        smu_inv.dot(pmu['mu']).reshape(nm,1))
    
    # these values are unused
    model = norm(loc=A.dot(pmu['mu']) + py['mu'], scale=py['sigma'] + A.dot(pmu['sigma']).dot(A.T))
    log_evidence = norm.pdf(y)
    return post, log_evidence

def sensor_fusion():

    sigmas = [0.01 * np.eye(2), 0.01*np.eye(2)]
    helper(sigmas)
    pml.savefig("demoGaussBayes2dEqualSpherical.pdf")
    plt.show()
    
    sigmas = [ 0.05*np.eye(2), 0.01*np.eye(2) ]
    helper(sigmas)
    pml.savefig("demoGaussBayes2dUnequalSpherical.pdf")
    plt.show()

    sigmas = [0.01*np.array([[10, 1], [1, 1]]), 0.01*np.array([[1, 1], [1, 10]])]
    helper(sigmas)
    pml.savefig("demoGaussBayes2dUnequal.pdf")
    plt.show()

def helper(sigmas):
    #initialization
    y1 = np.array([0, -1]).T
    y2 = np.array([1, 0]).T
    y = np.hstack((y1, y2))

    #use dictionary throughout for consistency
    prior = {}
    prior['mu'] = np.array([0,0]).T
    prior['sigma'] = 1e10 * np.eye(2)
    A = np.tile(np.eye(2), (2,1))

    py = {}
    py['mu'] = np.zeros((4,1))
    py['sigma'] = block_diag(sigmas[0], sigmas[1])

    post, log_evidence = gauss_soft_condition(prior, py, A, y)
    gauss_plot2d(y1, sigmas[0], {"color":"r"})
    gauss_plot2d(y2, sigmas[1], {"color":"g"})
    gauss_plot2d(post['mu'], post['sigma'],{"color":"k"})

if __name__ == "__main__":
    sensor_fusion()

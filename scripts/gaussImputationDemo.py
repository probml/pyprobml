# Author: Meduri Venkata Shivaditya
# Illustration of data imputation using an MVN.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix

def is_pos_def(x):
    #Check if the matrix is positive definite
    j = np.linalg.eigvals(x)
    return np.all(j>0)

def gauss_sample(mu, sigma, n):
    # Returns n samples (in the rows) from a multivariate Gaussian distribution

    a = np.linalg.cholesky(sigma)
    z = np.random.randn(len(mu), n)
    k = np.dot(a, z)
    return np.transpose(mu + k)

def gauss_condition(mu, sigma, visible_nodes, visible_values):
    # p(xh | xv = visValues)

    d = len(mu)
    j = np.array(range(d))
    v = visible_nodes.reshape(len(visible_nodes))
    h = np.setdiff1d(j, v)
    if len(h)==0:
        mugivh = np.array([])
        sigivh = np.array([])
    elif len(v) == 0:
        mugivh = mu
        sigivh = sigma
    else:
        ndx_hh = np.ix_(h, h)
        sigma_hh = sigma[ndx_hh]
        ndx_hv = np.ix_(h, v)
        sigma_hv = sigma[ndx_hv]
        ndx_vv = np.ix_(v, v)
        sigma_vv = sigma[ndx_vv]
        sigma_vv_inv = np.linalg.inv(sigma_vv)
        visible_values_len = len(visible_values)
        mugivh = mu[h] + np.dot(sigma_hv, np.dot(sigma_vv_inv, (visible_values.reshape((visible_values_len,1))-mu[v].reshape((visible_values_len, 1)))))
        sigivh = sigma_hh - np.dot(sigma_hv, np.dot(sigma_vv_inv, np.transpose(sigma_hv)))
    return mugivh, sigivh

def gauss_impute(mu, sigma, x):
    #Perform Gauss Imputation to the matrix x using mu and sigma
    #Fill in NaN entries of X using posterior mode on each row
    #Xc(i,j) = E[X(i,j) | D]

    n_data, data_dim = x.shape
    x_imputed = np.copy(x)
    for i in range(n_data):
        hidden_nodes = np.argwhere(np.isnan(x[i, :]))
        visible_nodes = np.argwhere(~np.isnan(x[i, :]))
        visible_values = np.zeros(len(visible_nodes))
        for tc, h in enumerate(visible_nodes):
            visible_values[tc] = x[i, h]
        mu_hgv, sigma_hgv = gauss_condition(mu, sigma, visible_nodes, visible_values)
        for rr, h in enumerate(hidden_nodes):
            x_imputed[i, h] = mu_hgv[rr]
    return x_imputed


def hinton_diagram(matrix, max_weight=None, ax=None, pl = None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else pl.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('white')
    ax.set_aspect('equal', 'box')

    for (x, y), w in np.ndenumerate(matrix):
        color = 'lawngreen' if w > 0 else 'royalblue'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    ax.grid(linestyle='--')
    ax.autoscale_view()
    ax.invert_yaxis()

def main():
    np.random.seed(12)
    data_dim = 8
    n_data = 10
    threshold_missing = 0.5
    mu = np.random.randn(data_dim, 1)
    sigma = make_spd_matrix(n_dim=data_dim)  # Generate a random positive semi-definite matrix
    # test if the matrix is positive definite
    # print(is_pos_def(sigma))
    x_full = gauss_sample(mu, sigma, n_data)
    missing = np.random.rand(n_data, data_dim) < threshold_missing
    x_miss = np.copy(x_full)
    x_miss[missing] = np.nan
    x_imputed = gauss_impute(mu, sigma, x_miss)
    #Create a matrix from x_miss by replacing the NaNs with 0s to display the hinton_diagram
    xmiss0 = np.copy(x_miss)
    for g in np.argwhere(np.isnan(x_miss)):
        xmiss0[g[0], g[1]] = 0
    plot_1 = plt.figure(1)
    hinton_diagram(xmiss0, pl=plot_1)
    plot_1.suptitle('Observed')
    plot_1.savefig("Hinton_Observed.png", dpi=300)
    plot_2 = plt.figure(2)
    hinton_diagram(x_full, pl=plot_2)
    plot_2.suptitle('Truth')
    plot_2.savefig("Hinton_Truth.png", dpi=300)
    plot_3 = plt.figure(3)
    hinton_diagram(x_imputed, pl=plot_3)
    plot_3.suptitle('imputation with true params')
    plot_3.savefig("Hinton_ImputationWithTrueParams.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
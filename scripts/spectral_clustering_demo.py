import superimport

import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import pyprobml_utils as pml

plt.style.use('classic')

def spectral_clustering_demo():
    np.random.seed(0)
    num_clusters = 2
    for data_type, data in (('circle', sample_circle(num_clusters)),
                            ('spiral', sample_spiral())):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(data)
        assignments = kmeans.predict(data)
        plot_data(data, assignments, 'k-means clustering', data_type)

        sigma = 0.1
        gamma = 1 / (2 * sigma ** 2)
        W = rbf_kernel(data, gamma=gamma)
        d = np.sum(W, 1, keepdims=True)
        sqrt_d = np.sqrt(d)

        normalized_W = (W / sqrt_d) / sqrt_d.T
        paranoid_assert(W, normalized_W, False)

        # We select the largest eigen values of normalized_W, rather
        # than the smallest eigenvalues of I - normalized_W.  The two
        # problems are equivalent. The eigen values can be converted
        # between the two problems via `1 - eigen_values`. The eigen
        # vectors are the same between both problems.
        eigen_values, eigen_vectors = eigh(normalized_W,
                                           # Get only the top num_clusters eigenvalues
                                           eigvals=(data.shape[0] - num_clusters, data.shape[0]-1))
        eigen_vectors = eigen_vectors / np.linalg.norm(eigen_vectors, axis=1, keepdims=True)

        kmeans.fit(eigen_vectors)
        assignments = kmeans.predict(eigen_vectors)
        plot_data(data, assignments, 'spectral clustering', data_type)

        plt.show()

def paranoid_assert(W, normalized_W, enable):
    if not enable:
        return
    D = np.diag(np.sum(W, 1))
    L = D - W
    D_inv_sqrt = np.diag(1 / np.diag(np.sqrt(D)))
    np.testing.assert_almost_equal(np.sum(L, 1), 0, err_msg="Rows of Laplacian must sum to 0.")
    np.testing.assert_allclose(normalized_W, D_inv_sqrt * W * D_inv_sqrt, rtol=0, atol=1)

def sample_circle(num_clusters):
    points_per_cluster = 500
    bandwidth = 0.1

    data = np.zeros((num_clusters * points_per_cluster, 2))
    for k, n in itertools.product(range(num_clusters), range(points_per_cluster)):
        theta = 2 * np.pi * np.random.uniform()
        rho = k + 1 + np.random.randn() * bandwidth
        x, y = pol2cart(theta, rho)
        idx = k * points_per_cluster + n
        data[idx, 0] = x
        data[idx, 1] = y
    data = data.reshape((num_clusters * points_per_cluster, 2))
    return data

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)

def sample_spiral():
    # Only 2 clusters in this case. This is hard-coded.
    points_per_cluster = 500
    bandwidth = 0.1

    data = np.empty((points_per_cluster, 2))

    w = np.arange(1, points_per_cluster + 1).astype(np.float32) / points_per_cluster
    data[:,0] = (4 * w + 1) * np.cos(2*np.pi * w) + np.random.randn(points_per_cluster) * bandwidth
    data[:,1] = (4 * w + 1) * np.sin(2*np.pi * w) + np.random.randn(points_per_cluster) * bandwidth
    data = np.vstack((data, -data))

    return data

def plot_data(data, assignments, title, data_type):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(data[assignments == 0, 0], data[assignments == 0, 1], 'o', color='r')
    ax.plot(data[assignments == 1, 0], data[assignments == 1, 1], 'o', color='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('square')
    ax.grid(True)
    ax.set_title(title)
    plt.tight_layout()
    pml.savefig(f"{data_type}_{title.replace(' ', '_')}.pdf")

if __name__ == '__main__':
    spectral_clustering_demo()

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.kernel_approximation import RBFSampler


# Relevance Vector Machine Classifier using EM algorithm by Michael E. Tipping.
### This is a python implementation of Relevance Vector Machine Classifier, it's based on github.com/ctgk/PRML/blob/master/prml/kernel/relevance_vector_classifier.py
class RVC:
    def sigmoid(self,a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    # Kernel matrix using rbf kernel with gamma = 0.3.
    def kernel_mat(self,X, Y):
        (x, y) = (np.tile(X, (len(Y), 1, 1)).transpose(1, 0, 2),
                  np.tile(Y, (len(X), 1, 1)))
        d = np.repeat(1 / (0.3 * 0.3), X.shape[-1]) * (x - y) ** 2
        return np.exp(-0.5 * np.sum(d, axis=-1))
    def __init__(self, alpha=1.):
        self.threshold_alpha = 1e8
        self.alpha = alpha
        self.iter_max = 100
        self.relevance_vectors_ = []

    # estimates for singulat matrices.
    def ps_inv(self, m):
        # assuming it is a square matrix.
        a = m.shape[0]
        i = np.eye(a, a)
        return np.linalg.lstsq(m, i, rcond=None)[0]

    '''
    For the current fixed values of alpha, the most probable
    weights are found by maximizing w over p(w/t,alpha) 
    using the Laplace approximation of finding an hessian.
    (E step)
    w = mean of p(w/t,alpha)
    cov = negative hessian of p(w/t,alpha)
    
    '''
    def _map_estimate(self, X, t, w, n_iter=10):
        for _ in range(n_iter):
            y = self.sigmoid(X @ w)
            g = X.T @ (y - t) + self.alpha * w
            H = (X.T * y * (1 - y)) @ X + np.diag(self.alpha)  # negated Hessian of p(w/t,alpha)
            w -= np.linalg.lstsq(H, g, rcond=None)[0]  # works even if for singular matrices.
        return w, self.ps_inv(H)        # inverse of H is the covariance of the gaussian approximation.

    '''
    Fitting of input-target pairs works by
    iteratively finding the most probable weights(done by _map_estimate method)
    and optimizing the hyperparameters(alpha) until there is no
    siginificant change in alpha.
    
    (M step)
    Optimizing alpha:
        For the given targets and current variance(sigma^2) alpha is optimized over p(t/alpha,variance)
        It is done by Mackay approach(ARD).
        alpha(new) = gamma/mean^2
        where gamma = 1 - alpha(old)*covariance.
    
    After finding the hyperparameters(alpha),
    the samples which have alpha less than the threshold(hence weight >> 0)
    are choosen as relevant vectors.
    
    Now predicted y = sign(phi(X) @ mean) ( mean contains the optimal weights)
    '''
    def fit(self, X, y):
        Phi = self.kernel_mat(X, X)
        N = len(y)
        self.alpha = np.zeros(N) + self.alpha
        mean = np.zeros(N)
        for i in range(self.iter_max):
            param = np.copy(self.alpha)
            mean, cov = self._map_estimate(Phi, y, mean, 10)
            gamma = 1 - self.alpha * np.diag(cov)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 1e10, out=self.alpha)
            if np.allclose(param, self.alpha):
                break

        ret_alpha = self.alpha < self.threshold_alpha
        self.relevance_vectors_ = X[ret_alpha]
        self.y = y[ret_alpha]
        self.alpha = self.alpha[ret_alpha]
        Phi = self.kernel_mat(self.relevance_vectors_, self.relevance_vectors_)
        mean = mean[ret_alpha]
        self.mean, self.covariance = self._map_estimate(Phi, self.y, mean, 100)


    # gives probability for target to be class 0.
    def predict_proba(self, X):
        phi = self.kernel_mat(X, self.relevance_vectors_)
        mu_a = phi @ self.mean
        var_a = np.sum(phi @ self.covariance * phi, axis=1)
        return 1 - self.sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))

    def predict(self, X):
        phi = self.kernel_mat(X, self.relevance_vectors_)
        return (phi @ self.mean > 0).astype(np.int)


# scipy.io loadmat doesn't seem to be accept the version of this data file
data = {}
with h5py.File('/pyprobml/data/bishop2class.mat', 'r') as f:
    for name, d in f.items():
        data[name] = np.array(d)

X = data['X'].transpose()
Y = data['Y']
y = Y.flatten()
y = y - 1  # changing to {0,1}

# Feature Mapping X to rbf_features to simulate non-linear logreg using linear ones.
rbf_feature = RBFSampler(gamma=0.3, random_state=1)
X_rbf = rbf_feature.fit_transform(X)

# Using CV to find SVM regularization parameter.
C = np.power(2, np.linspace(-5, 5, 10))
mean_scores = [cross_val_score(SVC(kernel='rbf', gamma=0.3, C=c), X, y, cv=5).mean() for c in C]
c = C[np.argmax(mean_scores)]

classifiers = {
    'logregL2': LogisticRegression(C=0.2, penalty='l2',
                                   solver='saga',
                                   multi_class='ovr',
                                   max_iter=10000),
    'logregL1': LogisticRegression(C=1, penalty='l1',
                                   solver='saga',
                                   multi_class='ovr',
                                   max_iter=10000),
    'RVM': RVC(),
    'SVM': SVC(kernel='rbf', gamma=0.3, C=c, probability=True)
}

h = 0.05  # step size in the mesh

# Mesh to use in the boundary plotting.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


def plot_scatters(X, y):
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # creating scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired', marker='X', s=30)


def plot_SVs(SV):
    plt.scatter(SV[:, 0], SV[:, 1], s=100, facecolor="none", edgecolor="green")


for (name, clf) in classifiers.items():

    if name == 'logregL2':
        clf.fit(X_rbf, y)
        Z = clf.predict_proba(rbf_feature.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z[:, 0].reshape(xx.shape)
        plt.title(name + ", nerr= {}".format(np.sum(y != clf.predict(X_rbf))))
        plt.contour(xx, yy, Z, np.linspace(0, 1, 5), colors=['black', 'w'])
        plot_scatters(X, y)
        plt.show()
        plt.savefig("../figures/kernelBinaryClassifDemo{}.pdf".format(name),  dpi=300)

    elif name == 'logregL1':
        clf.fit(X_rbf, y)
        Z = clf.predict_proba(rbf_feature.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z[:, 0].reshape(xx.shape)
        plt.title(name + ", nerr= {}".format(np.sum(y != clf.predict(X_rbf))))
        plt.contour(xx, yy, Z, np.linspace(0, 1, 5), colors=['w','black', 'w'])
        plot_scatters(X, y)
        conf_scores = np.abs(clf.decision_function(X_rbf))
        SV = X[(conf_scores > conf_scores.mean())]  # samples having a higher confidence scores are taken as support vectors.
        plot_SVs(SV)
        plt.show()
        plt.savefig("../figures/kernelBinaryClassifDemo{}.pdf".format(name),  dpi=300)

    elif name == 'RVM':
        clf.fit(X, y)
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.title(name + ", nerr= {}".format(np.sum(y != clf.predict(X))))
        plt.contour(xx, yy, Z, np.linspace(0, 1, 5), colors=['black', 'w'])
        plot_scatters(X, y)
        plot_SVs(clf.relevance_vectors_)
        plt.show()
        plt.savefig("../figures/kernelBinaryClassifDemo{}.pdf".format(name),  dpi=300)

    elif name == 'SVM':
        clf.fit(X, y)
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 0]
        Z = Z.reshape(xx.shape)
        plt.title(name + ", nerr= {}".format(np.sum(y != clf.predict(X))))
        plt.contour(xx, yy, Z, colors=['w', 'w', 'w', 'black'])
        plot_scatters(X, y)
        plot_SVs(clf.support_vectors_)
        plt.show()
        plt.savefig("../figures/kernelBinaryClassifDemo{}.pdf".format(name),  dpi=300)

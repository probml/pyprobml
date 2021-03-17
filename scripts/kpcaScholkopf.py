import numpy as np
import matplotlib.pyplot as plt


'''
% parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

np.random.seed(1)
rbf_var = 0.1
xnum = 4
ynum = 2
max_ev = xnum * ynum
#  % (extract features from the first <max_ev> Eigenvectors)
x_test_num = 15
y_test_num = 15
cluster_pos = np.array([[-0.5, -0.2], [0, 0.6], [0.5, 0]])
cluster_size = 30
'''
% generate a toy data set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
num_clusters = cluster_pos.shape[0]
train_num = num_clusters * cluster_size
patterns = np.zeros((train_num, 2))
rang = 1
for i in range(0, num_clusters):
    patterns[i * cluster_size:(i + 1) * cluster_size, 0] = cluster_pos[i, 0] + 0.1 * np.random.randn(cluster_size,
                                                                                                     1).reshape((30,))
    patterns[i * cluster_size:(i + 1) * cluster_size, 1] = cluster_pos[i, 1] + 0.1 * np.random.randn(cluster_size,
                                                                                                     1).reshape((30,))
test_num = x_test_num * y_test_num
x_range = np.linspace(-rang, rang, x_test_num)
y_offset = 0.5
y_range = np.linspace(-rang + y_offset, rang + y_offset, y_test_num)
xs, ys = np.meshgrid(x_range, y_range)
test_patterns = np.zeros((225, 2))
test_patterns[:, 0] = xs.flatten()
test_patterns[:, 1] = ys.flatten()
cov_size = train_num  #  % use all patterns to compute the covariance matrix
'''
% carry out Kernel PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
K = np.zeros((cov_size, cov_size))
for i in range(0, cov_size):
    for j in range(i, cov_size):
        K[i, j] = np.exp(-np.square(np.linalg.norm(patterns[i, :] - patterns[j, :])) / rbf_var)
        K[j, i] = K[i, j]


unit = np.ones((cov_size, cov_size)) / cov_size
#  % centering in feature space!
K_n = K - unit * K - K * unit + unit * (K * unit)

evals, evecs = np.linalg.eig(K_n)
evals[evals<0] = 0
evals = np.real(evals)

for i in range(0, cov_size):
    if evals[i] == 0:
        evecs[:, i] = np.inf
    else:
        evecs[:, i] = evecs[:, i] / (np.sqrt(evals[i]))

'''
% extract features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
unit_test = np.ones((test_num, cov_size)) / cov_size
K_test = np.zeros((test_num, cov_size))
for i in range(0, test_num):
    for j in range(0, cov_size):
        K_test[i, j] = np.exp(-np.square(np.linalg.norm(test_patterns[i, :] - patterns[j, :])) / rbf_var)

K_test_n = K_test - np.matmul(unit_test, K) - np.matmul(K_test, unit) + np.matmul(unit_test, K * unit)
test_features = np.matmul(K_test_n, evecs[:, 0:max_ev])
'''
% plot it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

for n in range(0, max_ev):
    ax = plt.subplot(ynum, xnum, n+1)
    plt.axis([-rang, rang, -rang + y_offset, rang + y_offset])
    imag = test_features[:, n].reshape(y_test_num, x_test_num)
    plt.plot(patterns[:, 0], patterns[:, 1], 'r.',markersize=2)
    X, Y = np.meshgrid(x_range, y_range)
    plt.contour(X,Y,imag,colors=['blue'],levels=9)
    ax.set_title("Eigenvalue={0:.4f}".format(evals[n]),fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    plt.tight_layout()

plt.savefig("/pyprobml/figures/kpcaScholkopfNoShade.pdf",  dpi=300)
plt.show()


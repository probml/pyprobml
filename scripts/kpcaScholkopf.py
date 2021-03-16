import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.lib import math
np.set_printoptions(threshold=sys.maxsize)
rbf_var = 0.1
xnum = 4
ynum = 2
max_ev = xnum*ynum
x_test_num = 15
y_test_num = 15
cluster_pos = np.array([[-0.5, -0.2], [0, 0.6], [0.5, 0]])
cluster_size = 30

num_clusters = cluster_pos.shape[0]
train_num = num_clusters*cluster_size
patterns = np.zeros([train_num, 2])
Range = 1

np.random.seed(0)
# print(num_clusters)
for i in range(3):
    for j in range((i-1)*cluster_size, i*cluster_size):
        patterns[j, 0] = cluster_pos[i, 0]+0.1*np.random.randn()
        patterns[j, 1] = cluster_pos[i, 1]+0.1*np.random.randn()

test_num = x_test_num*y_test_num
x_range = np.arange(-Range, Range+(2*Range/(x_test_num - 1)),
                    (2*Range/(x_test_num - 1)))
y_offset = 0.5
y_range = np.arange(-Range + y_offset, Range + y_offset +
                    (2*Range/(y_test_num - 1)), (2*Range/(y_test_num - 1)))
xs, ys = np.meshgrid(x_range, y_range)
test_patterns = np.zeros((225, 2))
a = 0
for i in range(0, 15):
    for j in range(0, 15):
        test_patterns[a, 0] = xs[j, i]
        test_patterns[a, 1] = ys[j, i]
        a += 1

cov_size = train_num
K = np.zeros((cov_size, cov_size))

for i in range(0, cov_size):
    for j in range(i, cov_size):
        K[i, j] = np.exp(-math.pow(np.linalg.norm(patterns[i,
                                                           :] - patterns[j, :]), 2)/rbf_var)
        K[j, i] = K[i, j]

unit = np.ones((cov_size, cov_size))/cov_size
K_n = K - unit*K - K*unit + unit*K*unit
evals, evecs = np.linalg.eig(K_n)
for i in range(0, cov_size):
    evecs[:, i] = evecs[:, i]/(np.sqrt(evals[i]))

unit_test = np.ones((test_num, cov_size))/cov_size
K_test = np.zeros((test_num, cov_size))


for i in range(0, test_num):
    for j in range(0, cov_size):
        for k in range(0, patterns.shape[1]):
            K_test[i, j] = np.exp(- math.pow(np.linalg.norm(
                test_patterns[i, k]-patterns[j, k]), 2)/rbf_var)


K_test_n = np.subtract(K_test, np.subtract(np.dot(unit_test, K), np.add(
    np.dot(K_test, unit),  np.dot(unit_test, np.dot(K, unit)))))
test_features = np.zeros((test_num, max_ev))
test_features = np.dot(K_test_n, evecs[:, 0:max_ev])
for n in range(0, max_ev):
    ax = plt.subplot(ynum, xnum, n+1)
    imag = np.reshape(test_features[:, n], (y_test_num, x_test_num))
    ax.contour(x_range, y_range, imag)
    #plt.plot(patterns[:, 0], patterns[:, 1], 'r.')
    #plt.text(-1, 1.65, "Eigenvalue="+str(round(evals[n], 3)))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.show()

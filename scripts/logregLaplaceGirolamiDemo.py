import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

if os.path.isdir('scripts'):
    os.chdir('scripts')
np.random.seed(0)
fs = 14
N = 30
D = 2
mu1 = np.concatenate((np.ones((N, 1)), 5*np.ones((N, 1))), axis=1)
mu2 = np.concatenate((-5*np.ones((N, 1)), 1*np.ones((N, 1))), axis=1)
class1_std = 1
class2_std = 1.1
X = np.concatenate((class1_std*np.random.randn(N, 2)+mu1,
                    2*class2_std*np.random.randn(N, 2)+mu2))
t = np.concatenate((np.ones((N, 1)), np.zeros((N, 1))))
alpha = 100

Range = 8
Step = 0.1
w1, w2 = np.meshgrid(np.arange(-Range, Range+Step, Step),
                     np.arange(-Range, Step+Range, Step))
n = w1.shape[0]
W = np.concatenate((w1.reshape((n*n, 1)), w2.reshape((n*n, 1))), axis=1)
plt.plot(X[(np.argwhere(t == 1)[:, 0]), 0],
         X[(np.argwhere(t == 1)[:, 0]), 1], 'r.')
plt.plot(X[(np.argwhere(t == 0)[:, 0]), 0],
         X[(np.argwhere(t == 0)[:, 0]), 1], 'bo')

Xgrid = W
ws = np.array([[3, 1], [4, 2], [5, 3], [7, 3]])

for ii in range(0, ws.shape[0]):
    w = np.transpose(ws[ii, :])
    pred = 1/(1+np.exp(-np.dot(Xgrid, w)))
    cc = plt.contour(w1, w2, pred.reshape((n, n)), 1)

plt.savefig('../figures/logregLaplaceGirolamiData.png')
plt.show()

for ii in range(0, ws.shape[0]):
    w = np.transpose(ws[ii, :])
    print(np.linalg.norm(w))

f = np.dot(W, np.transpose(X))
Log_Prior = np.log(multivariate_normal.pdf(W, cov=np.eye(D)*alpha))
Log_Like = np.dot(np.dot(W, np.transpose(X)), t) - \
    np.sum(np.log(1+np.exp(f)), 1).reshape(((25921, 1)))


plt.contour(w1, w2, (-Log_Prior).reshape((n, n)), 30)
plt.title('Log-Prior', fontdict={'size': fs})
plt.savefig('../figures/logregLaplaceGirolamiPrior.png')
plt.show()


plt.contour(w1, w2, (-Log_Like).reshape((n, n)), 30)
plt.title('Log-Likelihood', fontdict={'size': fs})

for ii in range(0, ws.shape[0]):
    w = np.transpose(ws[ii, :])
    plt.annotate(str(ii+1), xy=(w[0], w[1]))

j = np.argmax(Log_Like)
i = Log_Like[j][0]
wmle = W[j, :]
x2 = 8
y2 = 3.3
plt.plot([0,  x2], [0, y2])
plt.grid()
plt.savefig('../figures/logregLaplaceGirolamiNLL.png')
plt.show()

Log_Prior = Log_Prior.reshape((25921, 1))
Log_Joint = np.add(Log_Like, Log_Prior)
plt.contour(w1, w2, (-Log_Joint).reshape((n, n)), 30)
plt.title('Log-Unnormalised Posterior', fontdict={'size': fs})
j = np.argmax(Log_Joint)
i = Log_Joint[j][0]
plt.plot(W[j, 0], W[j, 1], '.', markersize=40)
plt.grid()
plt.savefig('../figures/logregLaplaceGirolamiPost.png')
plt.show()

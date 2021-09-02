# Plot 2d NLL loss surface for  binary logistic regression with 1 feature
# Loosely based on 
# https://peterroelants.github.io/posts/neural-network-implementation-part02/

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from mpl_toolkits.mplot3d import axes3d, Axes3D 

np.random.seed(0)

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import datasets


iris = datasets.load_iris()

X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

log_reg = LogisticRegression(solver="lbfgs",  fit_intercept=True, penalty='none')
log_reg.fit(X, y)

w_mle = log_reg.coef_[0][0] # 12.947270212450366
b_mle = log_reg.intercept_[0]  # -21.125250539711022
ypred = log_reg.predict_proba(X)

# Add column of 1s to end of X to capture bias term
N = X.shape[0]
ones = np.ones((N,1))
X1 = np.hstack((X, ones))

log_reg1 = LogisticRegression(solver="lbfgs", fit_intercept=False,  penalty='none')
log_reg1.fit(X1, y)

w_mle1 = log_reg1.coef_[0][0] 
b_mle1 = log_reg1.coef_[0][1] 
ypred1 = log_reg1.predict_proba(X1)

assert np.isclose(w_mle, w_mle1)
assert np.isclose(b_mle, b_mle1)
assert np.isclose(ypred[0], ypred1[0]).all()


# Define the logistic function
def logistic(z): 
    return 1. / (1 + np.exp(-z))

# Define the prediction function y = 1 / (1 + numpy.exp(-x*w))
def predict_prob(x, w): 
    return logistic(x.dot(w.T))

    
# Define the NLL loss function (y=probability, t=binary target)
def loss(y, t):
    return - np.mean(
        np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

params =np.asmatrix([[w_mle, b_mle]])
ypred2 = predict_prob(X1,params)
#assert np.isclose(ypred1[:,1], ypred2).all()

# We compute the loss on a grid of (w, b) values.
# We use for loops for simplicity.
ngrid = 50
sf = 0.5
ws = np.linspace(-sf*w_mle, +sf*w_mle, ngrid)
bs = np.linspace(-sf*b_mle, +sf*b_mle, ngrid)
grid_w, grid_b = np.meshgrid(ws, bs)
loss_grid = np.zeros((ngrid, ngrid))
for i in range(ngrid):
    for j in range(ngrid):
        params = np.asmatrix([grid_w[i,j], grid_b[i,j]])
        p = predict_prob(X1, params)
        loss_grid[i,j] = loss(p, y)
        

# Plot the loss function surface
plt.figure()
plt.contourf(grid_w, grid_b, loss_grid, 20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('NLL', fontsize=12)
plt.xlabel('$w$', fontsize=12)
plt.ylabel('$b$', fontsize=12)
plt.title('Loss function surface')
pml.savefig('logregIrisLossHeatmap.pdf')
plt.show()

fig,ax = plt.subplots()
CS = plt.contour(grid_w, grid_b, loss_grid,  cmap='jet')
#plt.plot(b_mle, w_mle, 'x') # Plot centered at MLE
pml.savefig('logregIrisLossContours.pdf')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(grid_w, grid_b, loss_grid)
pml.savefig('logregIrisLossSurf.pdf')
plt.show()


# Demo of perceptron algorithm in 2d
# Code is modified from various sources, including 
#https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3
#http://stamfordresearch.com/scikit-learn-perceptron/
#https://glowingpython.blogspot.com/2011/10/perceptron.html
#https://stackoverflow.com/questions/31292393/how-do-you-draw-a-line-using-the-weight-vector-in-a-linear-perceptron?rq=1   
#https://medium.com/@thomascountz/calculate-the-decision-boundary-of-a-single-perceptron-visualizing-linear-separability-c4d77099ef38 
    

import superimport

import numpy as np
import matplotlib.pyplot as plt

figdir = "../figures"
def save_fig(fname):
    if figdir: plt.savefig(os.path.join(figdir, fname))
    
np.random.seed(0)


class Perceptron(object):

    def __init__(self, no_of_inputs, max_iter=10, learning_rate=1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        self.weights_hist = []
           
    def predict_single(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          output = 1
        else:
          output = 0            
        return output
    
    def predict(self, X):
        n = X.shape[0]
        yhat = np.zeros(n)
        for i in range(n):
            yhat[i] = self.predict_single(X[i,:])
        return yhat

    def fit(self, training_inputs, labels):
        for epoch in range(self.max_iter):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict_single(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                self.weights_hist.append(np.copy(self.weights))
            

def generateData(n):
     # Generates a 2D linearly separable dataset with 2n samples. 
     # Red and blue clusters in top left and bottom right quadrant
     xb = (np.random.rand(n)*2-1)/2-0.5
     yb = (np.random.rand(n)*2-1)/2+0.5
     xr = (np.random.rand(n)*2-1)/2+0.5
     yr = (np.random.rand(n)*2-1)/2-0.5
     XB = np.stack([xb,yb], axis=1)
     XR = np.stack([xr,yr], axis=1)
     X = np.concatenate([XB, XR])
     y = np.concatenate([np.zeros(n, dtype=np.int), np.ones(n, dtype=np.int)])
     return X, y


X, y = generateData(10)

    
def plot_dboundary_contourf(net):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    fig, ax = plt.subplots()
    Z = net.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z) #, cmap=plt.cm.Paired)
    ax.axis('off')
    colormap = np.array(['r', 'k'])
    ax.scatter(X[:, 0], X[:, 1], c=colormap[y])
    #ax.axis('square')

    
def plot_dboundary(weights, offset):
    w1 = weights[0]
    w2 = weights[1]
    b = offset
    slope = -w1/w2
    intercept = -b/w2
    xx = np.linspace(-1, 1, 10)
    yy = xx*slope + intercept
    plt.figure()
    colormap = np.array(['r', 'k'])
    plt.scatter(X[:,0], X[:,1], c=colormap[y], s=40)
    plt.plot(xx, yy, 'k-')


    
ninputs = 2
net = Perceptron(ninputs)
net.fit(X, y)

w = net.weights[1:]
b = net.weights[0]
plot_dboundary(w, b)
plot_dboundary_contourf(net)

'''
H = net.weights_hist
niter = len(H)
snapshots = [int(t) for t in np.linspace(0, 20, 5)]
#snapshots = [15,  20, 25]
for t in snapshots:
    w = H[t][1:]
    b = H[t][0]
    plot_dboundary(w, b)
    plt.title('iter {}'.format(t))
'''

# sklearn version
from sklearn.linear_model import Perceptron
net_sklearn = Perceptron()
net_sklearn.fit(X, y)
w = net_sklearn.coef_[0]
offset = net_sklearn.intercept_[0]
    
plot_dboundary(w, b)
plot_dboundary_contourf(net)



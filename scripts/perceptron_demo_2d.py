# Demo of perceptron algorithm in 2d
# Code is modified from various sources, including 
#https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3
#http://stamfordresearch.com/scikit-learn-perceptron/
#http://stamfordresearch.com/python-perceptron-re-visited/


import numpy as np
import matplotlib.pyplot as plt

figdir = "../figures"
def save_fig(fname):
    if figdir: plt.savefig(os.path.join(figdir, fname))
    
np.random.seed(42)


class Perceptron(object):

    def __init__(self, no_of_inputs, max_iter=20, learning_rate=1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        self.weights_hist = np.zeros((max_iter, no_of_inputs+1))
           
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
        for t in range(self.max_iter):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict_single(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                self.weights_hist[t,:] = self.weights
            

# Linearly separable data in 2d
X = np.array([
[2, 1, 2, 5, 7, 2, 3, 6, 1, 2, 5, 4, 6, 5],
[2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7]
])
X = X.T
y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])


# sklearn version
if 0:
    from sklearn.linear_model import perceptron
    net_sklearn = perceptron.Perceptron()
    net_sklearn.fit(X, y)
    w = net_sklearn.coef_[0]
    offset = net_sklearn.intercept_[0]
    
def plot_dboundary_contourf(net):
    # Plot decision boundary using contourf
    #https://stats.stackexchange.com/questions/71335/decision-boundary-plot-for-a-perceptron
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
    # Plot decision boundary by solving for the decision boundary
    #https://stackoverflow.com/questions/31292393/how-do-you-draw-a-line-using-the-weight-vector-in-a-linear-perceptron?rq=1   
    #https://medium.com/@thomascountz/calculate-the-decision-boundary-of-a-single-perceptron-visualizing-linear-separability-c4d77099ef38 
    w1 = weights[0]
    w2 = weights[1]
    b = offset
    slope = -w1/w2
    intercept = -b/w2
    
    xx = np.linspace(2, 5, 10)
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

'''
for t in range(20):
    w = net.weights_hist[t,1:]
    b = net.weights_hist[t,0]
    plot_dboundary(w, b)
    plt.title('iter {}'.format(t))
'''
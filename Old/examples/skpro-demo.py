
  
#https://alan-turing-institute.github.io/skpro/introduction.html
import sklearn
import skpro

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.datasets.base import load_boston
from sklearn.model_selection import train_test_split

from skpro.baselines import DensityBaseline
from skpro.metrics import log_loss

# Load boston housing data
X, y = load_boston(return_X_y=True) # X 506x13, y 506
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# X_train 354x13, X_test 152 x 13

# Train and predict on boston housing data using a baseline model
y_pred = DensityBaseline().fit(X_train, y_train)\
                          .predict(X_test)
# Obtain the loss
loss = log_loss(y_test, y_pred, sample=True, return_std=True)

print('Loss: %f+-%f' % loss)



def plot_performance(y_test, y_pred, filename=None):
    """
    Visualises the prediction performance

    Parameters
    ----------
    y_test  Ground truth
    y_pred  Predictions
    filename    If string, figure will be saved to file

    Returns
    -------
    Matplotlib plot
    """

    fig, ax1 = pyplot.subplots()

    ax1.plot(y_test, y_test, 'g.', label=u'Optimum')
    sigma = np.std(y_pred)
    ax1.errorbar(y_test, y_pred.point(), yerr=sigma, label=u'Predictions', fmt='b.', ecolor='r', linewidth=0.5)
    ax1.set_ylabel('Predicted $y_{pred}$')
    ax1.set_xlabel('Correct label $y_{true}$')
    ax1.legend(loc='best')

    losses = log_loss(y_test, y_pred, sample=False)
    ax2 = ax1.twinx()
    overall = "{0:.2f}".format(np.mean(losses)) + " +/- {0:.2f}".format(np.std(losses))
    ax2.set_ylabel('Loss')
    ax2.plot(y_test, losses, 'y_', label=u'Loss: ' + overall)
    ax2.tick_params(colors='y')
    ax2.legend(loc=1)

    if not isinstance(filename, str):
        pyplot.show()
    else:
        pyplot.savefig(filename, transparent=True, bbox_inches='tight')
        
# Plot performance
#import utils
#utils.
plot_performance(y_test, y_pred)

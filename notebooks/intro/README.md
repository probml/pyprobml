# Ch 1: Introduction

## What is machine learning?

Machine learning is a collection of methods for finding patterns in data.
In many cases, the desired "pattern"
is an output label associated with a particular input (piece
of data); a human provides examples of the kinds of input-output pairs he/she
is interested in,
and the algorithm  uses these labeled examples to learn an input-output mapping;
the learned mapping can be used to
predict labels (outputs) for new inputs that it has not seen before.
But we may also be interested in discovering "unknown patterns",
without having to specify any labels.
These may be clusters, or temporal or spatial trends, or dependencies
between various variables, etc.

To handle such a variety of tasks, we will need to consider a wide variety
of statistical models, and a wide variey of algorithms to fit
these models to data. Since there will always be uncertainty in what patterns
truly exist in the data, we will focus on probabilistic models; we can then
use decision theory to choose the best action, trading off the expected risk
of different outcomes.

All of this is explained in detail in the book.
In this notebook, we just include some pointers to some of the online code.
Most of the code is written in Python, which is the most
popular language for ML and datascience.
See [this link](https://github.com/probml/pyprobml/blob/master/notebooks/intro/software.md)
for an introduction to the rich Python ML software ecosystem.
See also [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/data.ipynb) for
an introduction to exploratory data analysis in Python.

 ## Supervised learning: foundations

 ### Optimization, overfitting, and cross validation
 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/optimization_overfitting_and_cross_validation.ipynb)


 ## Supervised learning: models

 ### Logistic regression <a class="anchor" id="logreg"></a>

 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/logreg.ipynb)

 ### Linear regression <a class="anchor" id="linreg"></a>

 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/linreg.ipynb)

 ### Deep neural networks <a class="anchor" id="DNN"></a>

 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/dnn1/dnn.ipynb)

 ### Bagging, boosting, trees and forests

 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/bagging_boosting_trees_and_forests.ipynb)

 ### K nearest neighbor (KNN) classification

 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/k_nearest_neighbor__classification.ipynb)

 ## Supervised learning: Regularization

 ### Bayesian machine learning

 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/svi_linear_regression_1d_tfp.ipynb)

 ### Other regularization techniques

 WIP

 ## Unsupervised learning <a class="anchor" id="unsuper"></a>

 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/unsuper.ipynb)

 ## Reinforcement learning

 WIP

 ## Figures

 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/figures/chapter1_figures.ipynb)

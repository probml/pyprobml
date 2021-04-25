# Python software ecosystem

## General coding

In this book, we will use Python 3.
For a good introduction, see e.g., the free books [Whirlwind tour of Python](https://github.com/jakevdp/WhirlwindTourOfPython)  by Jake Vanderplas
or [Dive into Python 3](https://www.diveinto.org/python3/table-of-contents.html) by Mark Pilgrim.

Each chapter is associated with one or more
 <a href="https://jupyter.org/">Jupyter notebooks</a>,
which mix code and results.
We use the [Google colab version of notebooks](https://colab.research.google.com/), which run on the Google Compute Platform (GCP),
so you don't have to install code locally.
To avoid installing packages every time you open a colab,
you can follow [these steps](https://stackoverflow.com/questions/55253498/how-do-i-install-a-library-permanently-in-colab).


When developing larger software projects locally, it is often better to use an 
 IDE (interactive development environment),
 which keeps the code separate from the results.
I like to use 
<a href="https://www.spyder-ide.org">Spyder</a>,
although many people use
<a href="https://www.jetbrains.com/pycharm/">PyCharm</a>.
For a browser-based IDE, you can use
<a href="https://github.com/jupyterlab/jupyterlab">JupyterLab</a>.

## Software for data science and ML

We will leverage many standard libraries from Python's "data science
stack", listed in the table below. 
For a good introduction to these, see e.g., the free book
[Python Datascience Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) by
Jake Vanderplas, 
or the class [Computational Statistics in Python](http://people.duke.edu/~ccc14/sta-663-2019/)  by Cliburn Chen at Duke University. For an excellent book on
[scikit-learn](https://scikit-learn.org/stable/), see
[Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow v2](https://github.com/ageron/handson-ml2) by Aurelion Geron. 



| Name | Functionality |
| ---- | ---- | 
| [Numpy](http://www.numpy.org) |  Vector and matrix computations |
| [Scipy](http://www.scipy.org) | Various scientific / math / stats / optimization functions   |
| [Matplotlib](http://matplotlib.org), [Seaborn](https://seaborn.pydata.org/) | Plotting |
| [Pandas](http://pandas.pydata.org), [Xarray](http://xarray.pydata.org/en/stable/index.html) | Dataframes and named arrays |
| [Scikit-learn](http://scikit-learn.org) | Many ML methods (excluding deep learning) |
| [Jax](http://github.com/google/jax) |  Accelerated version of Numpy with autograd support |


## Software for deep learning <a class="anchor" id="DL"></a>


Deep learning is about composing primitive differentiable functions
into a computation graph in order to make more
complex functions,  and then using
automatic differentiation ("autograd") to compute gradients of the
output with respect to model parameters , which we
can pass to an optimizer, to fit the function to data. This is
sometimes called "differentiable programming". 

DL therefore requires several different libraries,
to perform tasks such as

- specify the model
- compute gradients using automatic differentiation
- train the model (pass data to the gradient function,
and gradients to the optimizer function)
- serve the model (pass input to a trained model, and pass output
to some service)
 
The training and serving often uses 
hardware accelerators, such as GPUs. (Some libraries also support
distributed computation, but we will not need use this feature in this
book.)

There are several popular DL frameworks, which
implement the above functionality, some of as

|Name|More info|
|----|----|
|[Tensorflow2](http://www.tensorflow.org)|[tf_intro.ipynb](https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/tf_intro.ipynb)|
|[JAX](http://github.com/google/jax)|[JAX tutorials](https://github.com/probml/pyprobml/blob/master/notebooks/jax_tutorials.md)|
|[PyTorch](http://pytorch.org)|[PyTorch website](https://pytorch.org/tutorials)|
|[MXNet](https://mxnet.apache.org)|[Dive into deep learning book](http://www.d2l.ai)|

In this book, we mostly use Tensorflow 2 and JAX.
However, we also welcome contributions in PyTorch.
(More details on the JAX ecosystem can be found
[here](https://github.com/probml/pyprobml/blob/master/book1/intro/jax_ecosystem.md).)

        
## Software for probabilistic modeling <a class="anchor" id="PPL"></a>

In this book, we focus on probabilistic models, both
supervised (conditional) models of the form $$p(y|x)$$, as well as
unsupervised models of the form $$p(z,x)$$, where $$x$$ are the features,
$$y$$ are the labels (if present), and $$z$$ are the latent variables. For
simple special cases, such as GMMs and PCA, we can use
sklearn. However, to create more complex models, we need more flexible
libraries. We list some examples below.

The term  "probabilistic programming language" (PPL) is used to
describe systems that allow the creation of "randomly shaped" models,
whos structure is determined e.g., by stochastic control flow.  The
Stan library specifiis the model using a domain specific language
(DSL); most other libraries specify the model via an API. In this
book, we focus on PyMc3 and numpyro. 


|Name|Description|
|----|----|
|[Tensorflow probability](https://www.tensorflow.org/probability)|PPL built on top of TF2.|
|[Edward 1](http://edwardlib.org)|PPL built on top of TF2 or Numpy.|
|[Edward 2](https://github.com/google/edward2)|Low-level PPL built on top of TF2.|
|[Pyro](https://github.com/pyro-ppl/pyro)|PPL built on top of PyTorch.|
|[NumPyro](https://github.com/pyro-ppl/numpyro)|Similar to Pyro, but built on top of JAX instead of PyTorch.|
|[PyStan](https://pystan.readthedocs.io/en/latest)|Python interface to [Stan](https://mc-stan.org), which uses the BUGS DSL for PGMs. Custom C++ autodiff library.|
|[PyMc3](https://docs.pymc.io)|Similar to PyStan, but uses Theano for autodiff. (Future versions will use JAX.)|


There are also libraries for inference in probabilistic models
in which all variables are discrete. Such models do not need autograd.
We give some examples below. 

|Name|Description|
|----|----|
|[PgmPy](http://pgmpy.org/)|Discrete PGMs.|
|[Pomegranate](https://pomegranate.readthedocs.io/en/latest/index.html)|Discrete PGMs. GPU support.|


          




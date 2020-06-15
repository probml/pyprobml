# Supplement for Chapter "[Introduction](https://htmlpreview.github.io/?https://github.com/probml/pyprobml/blob/master/chapters/intro/intro.html)"

# Python software ecosystem

## General coding

In this book, we will use Python 3.
For a good introduction, see e.g., the free books [Whirlwind tour of Python](https://github.com/jakevdp/WhirlwindTourOfPython)  by Jake Vanderplas or [Dive into Python 3](https://www.diveinto.org/python3/table-of-contents.html) by Mark Pilgrim.

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

We will leverage many standard libraries from Python's "data science stack", listed in the table below.
For a good introduction to these, see e.g., the free book [Python Datascience Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) by Jake Vanderplas,
or the class [Computational Statistics in Python](http://people.duke.edu/~ccc14/sta-663-2019/)  by Cliburn Chen at Duke University. For an excellent book on [scikit-learn](https://scikit-learn.org/stable/), see [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow v2](https://github.com/ageron/handson-ml2) by Aurelion Geron.



| Name | Functionality |
| ---- | ---- | 
| [Numpy](http://www.numpy.org) |  Vector and matrix computations |
| [Jax](http://github.com/google/jax) |  Accelerated version of Numpy with autograd support (see [JAX notebook](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/jax.ipynb)) |
| [Scipy](http://www.scipy.org) | Various scientific / math / stats / optimization functions   |
| [Matplotlib](http://matplotlib.org), [Seaborn](https://seaborn.pydata.org/) | Plotting |
| [Pandas](http://pandas.pydata.org), [Xarray](http://xarray.pydata.org/en/stable/index.html) | Dataframes and named arrays |
| [Scikit-learn](http://scikit-learn.org) | Many ML methods (excluding deep learning) |

         
## Software for deep learning <a class="anchor" id="DL"></a>


Deep learning is about composing differentiable functions into more complex functions, represented as a computation graph, and then using automatic differentiation ("autograd") to compute gradients, which we can pass to an optimizer, to fit the function to data. This is sometimes called "differentiable programming".

There are several libraries that can execute such computation graphs on hardware accelerators, such as GPUs. (Some libraries also support distributed computation, but we will not need use this feature in this book.) We list a few popular libraries below.  In this book, we focus on [Flax](https://github.com/google/flax), a library built on top of [JAX](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/jax.ipynb), although we also use Tensorflow for some examples. (Note that JAX is a purely functional library, which differs from Tensorflow and  Pytorch; Flax provides some "magic" on top that makes it behave as it was stateful, which  makes it easier to implement complex DNNs. It is also possible to use raw JAX, which is just like using numpy, but faster, and with autograd.)  
     
     
 <table align="left">
    <tr>
        <th>Name</th>
      <th>More info</th>
    <tr> 
        <td> <a href="http://www.tensorflow.org">Tensorflow 2.0</a></td>
     <td><a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/tf.ipynb">TF notebook</a>
               <tr>
        <td> <a href="https://github.com/google/flax">FLAX</a>
            <td> <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/flax.ipynb"> Flax notebook</a>
    <tr>
        <td> <a href="http://pytorch.org">Pytorch</a>
       <td>
       <a href="https://pytorch.org/tutorials/">Official PyTorch tutorials</a>
              <tr>
        <td> <a href="https://mxnet.apache.org/">MXNet</a>
              <td>
                <a href="http://www.d2l.ai/">  Dive into deep learning book</a>       
</table>
        
## Software for probabilistic modeling <a class="anchor" id="PPL"></a>

In this book, we will be focusing on probabilistic models, both supervised (conditional) models of the form `$p(y|x)$`, as well as unsupervised models of the form $p(z,x)$, where $x$ are the features, $y$ are the labels (if present), and $z$ are the latent variables. For simple special cases, such as GMMs and PCA, we can use sklearn. However, to create more complex models, we need more flexible libraries. We list some examples below.
The term  "probabilistic programming language" (PPL) is used to describe systems that allow the creation of "randomly shaped" models, whos structure is determined e.g., by stochastic control flow.  The Stan library specifiis the model using a domain specific language (DSL); most other libraries specify the model via an API. In this book, we focus on PyMc3 and numpyro.


<table align="left">
<tr>
<th style="text-align:left">Name</th>
<th style="text-align:left" width="400">Functionality</th>
  <tr>
     <td style="text-align:left"> <a href="https://www.tensorflow.org/probability">TF Probability</a> (TFP)
         <td style="text-align:left"> PPL built on top of TF2.
    <tr>
      <td style="text-align:left"> <a href="http://edwardlib.org/">Edward</a> 
         <td style="text-align:left"> PPL built on top of TF2.
          <tr>
 <td style="text-align:left"> <a href="https://github.com/google/edward2">Edward 2</a> 
         <td style="text-align:left"> PPL built on top of TF2 or Numpy
    <tr>
    <td style="text-align:left"> <a href="https://github.com/pyro-ppl/pyro">Pyro</a>
<td  style="text-align:left"> PPL built on top of PyTorch.
<tr>
    <td style="text-align:left"> <a href="https://github.com/pyro-ppl/numpyro">NumPyro</a>
<td style="text-align:left"> Similar to Pyro, but built on top of JAX instead of PyTorch.
<tr>
     <td style="text-align:left"> <a href="https://pystan.readthedocs.io/en/latest/">PyStan</a>
    <td style="text-align:left"> Python interface to <a href="https://mc-stan.org">Stan</a>, which uses the BUGS DSL for PGMs. Custom C++ autodiff library.
              <tr>
     <td style="text-align:left"> <a href="https://docs.pymc.io/">PyMc</a>
         <td style="text-align:left"> Similar to PyStan, but uses Theano (v3) or TF (v4) for autograd.
 </table>


There are also libraries for inference in probabilistic models which do not need autograd, such as probabilistic graphical models with discrete latent variables. We give some examples below.

<table align="left">
 <tr>
     <td style="text-align:left"> <a href="http://pgmpy.org/">PgmPy</a>
         <td style="text-align:left">  Discrete PGMs. 
<tr>            
     <td style="text-align:left"> <a href="https://pomegranate.readthedocs.io/en/latest/index.html">Pomegranite</a>
        <td style="text-align:left"> Discrete PGMs. GPU support.
</table>
          
 # Exploratory data analysis <a class="anchor" id="EDA"></a>
 
 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/data.ipynb) 
 
 # Supervised learning
 
 ## Logistic regression <a class="anchor" id="logreg"></a>
 
 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/logreg.ipynb) 
 
 ## Linear regression <a class="anchor" id="linreg"></a>
 
 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/linreg.ipynb) 
 
 ## Deep neural networks <a class="anchor" id="DNN"></a>
 
 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/dnn/dnn.ipynb) 
 
 # Unsupervised learning <a class="anchor" id="unsuper"></a>
 
 See [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/unsuper.ipynb) 
 

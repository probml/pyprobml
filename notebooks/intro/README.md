# Supplement for Chapter "[Introduction](https://htmlpreview.github.io/?https://github.com/probml/pyprobml/blob/master/chapters/intro/intro.html)"

## Introduction

In this notebook, we introduce some of the software we will use in the book.

## Python

In this book, we will use Python 3.
For a good introduction, see e.g., the free books [Whirlwind tour of Python](https://github.com/jakevdp/WhirlwindTourOfPython)  by Jake Vanderplas or [Dive into Python 3](https://www.diveinto.org/python3/table-of-contents.html) by Mark Pilgrim.

Each chapter is associated with one or more
 <a href="https://jupyter.org/">Jupyter notebooks</a>,
which mix code and results.
We use the [Google colab version of notebooks](https://colab.research.google.com/), which run in the cloud,
so you don't have to install code locally.

When developing larger software projects, it is often better to use an 
 IDE (interactive development environment),
 which keeps the code separate from the results.
I like to use 
<a href="https://www.spyder-ide.org">Spyder</a>,
although many people use
<a href="https://github.com/jupyterlab/jupyterlab">JupyterLab</a>
for a browser-based solution.

## Software for deep learning <a class="anchor" id="DL"></a>


Deep learning is about composing differentiable functions into more complex functions, represented as a computation graph, and then using automatic differentiation ("autograd") to compute gradients, which we can pass to an optimizer, to fit the function to data. This is sometimes called "differentiable programming".

There are several libraries that can execute such computation graphs on hardware accelerators, such as GPUs. (Some libraries also support distributed computation, but we will not need use this feature in this book.) We list a few popular libraries below. The most popular (at the time of writing) are Tensorflow and Pytorch. The newest is JAX. This is fairly different in "flavor" since it is purely functional; it is also the most similar to regular numpy, making it excellent for research and teaching.



     
     
 <table align="left">
    <tr>
        <th>Name</th>
      <th>More info</th>
    <tr> 
        <td> <a href="http://www.tensorflow.org">Tensorflow 2.0</a></td>
     <td><a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/tf.ipynb">TF notebook</a>
               <tr>
        <td> <a href="http://github.com/google/jax">JAX</a>
            <td>
              <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/jax.ipynb">JAX notebook</a>
    <tr>
        <td> <a href="http://pytorch.org">Pytorch</a>
       <td>
       <a href="https://pytorch.org/tutorials/">Official PyTorch tutorials</a>
              <tr>
        <td> <a href="https://mxnet.apache.org/">MXNet</a>
              <td>
                <a href="http://www.d2l.ai/">  Dive into deep learning book</a>       
</table>

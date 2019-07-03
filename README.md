# pyprobml
Python 3 code for the second edition of my book "Machine learning: a probabilistic perspective" (http://people.cs.ubc.ca/~murphyk/MLbook/). This is work in progress, so expect rough edges.

 

## Installation

We assume you have installed numpy, scipy, matplotlib, seaborn, pandas, scikit-learn, etc.
(All of these are bundled with [anaconda](https://www.anaconda.com/distribution/).)

Some scripts rely on additional libraries, such as the following: 
- [tensorflow 2.0](https://www.tensorflow.org/)
- [tensorflow probability](https://www.tensorflow.org/probability)
- [jax](https://github.com/google/jax)
- [pytorch](https://pytorch.org/)

You also need to define the PYPROBML environment variable.
You can do this by adding the following
line to your .bash_profile file, and then starting a new shell.
```
    export PYPROBML="/Users/kpmurphy/github/pyprobml" # replace with your download location
```


## Scripts

Most of the code lives in the [scripts](scripts) directory. To execute a script, cd (change directory) to the scripts folder,
and then type 'python foo.py'. You can also run each script from inside a Python IDE (like Spyder).
Many of the scripts create plots, which are saved to PYROBML/figures.


## Notebooks

We have created notebooks for some of the chapters to show how to convert the theory in the book into practice. (Work in progress...)

* [Introduction](https://github.com/probml/pyprobml/blob/master/notebooks/introduction.ipynb)
* [Linear algebra](https://github.com/probml/pyprobml/blob/master/notebooks/linear_algebra.ipynb)
* [Probability](https://github.com/probml/pyprobml/blob/master/notebooks/probability.ipynb)
* [Optimization](https://github.com/probml/pyprobml/blob/master/notebooks/optimization.ipynb)
    



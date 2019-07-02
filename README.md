# pyprobml
Python 3 code for the second edition of my book "Machine learning: a probabilistic perspective" (http://people.cs.ubc.ca/~murphyk/MLbook/). This is work in progress, so expect rough edges.

 

## Installation

We assume you have installed numpy, scipy, matplotlib, seaborn, pandas, scikit-learn, etc.
(All of these are bundled with anaconda.)

Some scripts rely on additional libraries, such as the following: 
- tf 2.0: tensorflow (https://www.tensorflow.org/)
- tfp:  tensorflow probability (pip install --upgrade tfp-nightly)
- jax:   (https://github.com/google/jax)
- pymc3:  (https://docs.pymc.io/)
- pytorch:  (https://pytorch.org/)

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

We have created a few notebooks to illustrate some of the techniques in more detail.
These are listed below.

* Chapter 1 (Introduction)
    * [iris_explore](notebooks/iris_explore.ipynb).
    * [autompg_explore](notebooks/autompg_explore.ipynb).
    * (autompg_preprocessing)[notebooks/autompg_preprocessing.ipynb]. 
* Chapter 2 (Linear algebra)
    * Foo. Needs jax.
* Chapter 3 (Probabilty).
    * None yet.
* Chapter 4 (Probabilistic graphical models).
    * None yet.
    



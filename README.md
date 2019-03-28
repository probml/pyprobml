# pyprobml
Python 3 code for the second edition of my book "Machine learning: a probabilistic perspective" (http://people.cs.ubc.ca/~murphyk/MLbook/). This is work in progress, so expect rough edges.

The main code lives in pyprobml/book. You can execute any script from the command line
using 'python foo.py', or from inside a Python IDE (like Spyder). 

Many of the scripts create plots.
These are saved to the directory PYROBML/figures. To **set this environment variable**, add the following
line to your .bash_profile file before opening a new terminal:
```
    export PYPROBML="/Users/kpmurphy/github/pyprobml" # replace with your download location
```
If you don't set it, the plots won't be saved.
(However it is useful to set it anyway since some scripts also need to find PYPROBML/data.)

We assume you have installed numpy, scipy, matplotlib, pandas, scikit-learn.
(These are all pre-installed in anaconda.) 
Many of the scripts rely on extra libraries which you will need to install.
This will often be indicated in the suffix of the filename, as follows:
- tf or keras: tensorflow (https://www.tensorflow.org/)
- tfp:  tensorflow probability (pip install --upgrade tfp-nightly)
- jax:   (https://github.com/google/jax)
- pymc3:  (https://docs.pymc.io/)
- pytorch:  (https://pytorch.org/)



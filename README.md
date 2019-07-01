# pyprobml
Python 3 code for the second edition of my book "Machine learning: a probabilistic perspective" (http://people.cs.ubc.ca/~murphyk/MLbook/). This is work in progress, so expect rough edges.

The main code lives in [scripts](scripts). To execute a script, cd (change directory) to the scripts folder,
and then type 'python foo.py'. You can also run each script from inside a Python IDE (like Spyder). 

Many of the scripts create plots. 
These are saved to PYROBML/figures, where PYROBML is the root directory (parent of scripts).
Currently you have to define the PYPROBML environment variable before running the scripts.
You can do this by adding the following
line to your .bash_profile file, and then starting a new shell.
```
    export PYPROBML="/Users/kpmurphy/github/pyprobml" # replace with your download location
```

We assume you have installed numpy, scipy, matplotlib, pandas, scikit-learn.
Some scripts rely on additional libraries, such as the following: 
- tf or keras: tensorflow (https://www.tensorflow.org/)
- tfp:  tensorflow probability (pip install --upgrade tfp-nightly)
- jax:   (https://github.com/google/jax)
- pymc3:  (https://docs.pymc.io/)
- pytorch:  (https://pytorch.org/)



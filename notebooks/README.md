# Notebooks

## Instructions for contributors

### How to contribute?

It is recommended to use Python `3.7.13` because our automated GitHub workflow uses `3.7.13` to check the code and currently Google colab Python version is also `3.7.13`. You may use [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install a specific version of Python on your system.

1. Choose a figure from [book1](https://probml.github.io/pml-book/book1.html) ([pdf](https://github.com/probml/pml-book/releases/latest/download/book1.pdf)) or [book2](https://probml.github.io/pml-book/book2.html) ([pdf](https://github.com/probml/pml2-book/releases/latest/download/pml2.pdf)) and find out its source code notebook in this repo (mostly in the `notebooks` folder).
2. Fork the main repo and create a new branch with a meaningful name.
3. Take [this notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book1/02/discrete_prob_dist_plot.ipynb) as a reference while converting the source code to a notebook. In particular, follow these steps:
    * Wrap imports in `try: except:` blocks. for example:
    ```python
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        %pip install -qq tensorflow
        import tensorflow as tf

    from tensorflow import keras
    ```

    * Do not wrap imports in `try: except:` blocks if a package is present in [requirements.txt](requirements.txt) or it is an inbuilt python package such as `os`, `sys`, `functools` etc.
    * Note that `latexify` function will be effective only if "LATEXIFY" is set in the environment. For more details check "Generating figures and saving them locally" section below.
    * Set appropriate height (you can refer exact height of figures from here: [book1](https://github.com/probml/pyprobml/blob/master/internal/fig_height/fig_height_book1.md) & [book2](https://github.com/probml/pyprobml/blob/master/internal/fig_height/fig_height_book2.md))  and width using the latexify function. for example: 
    ```py
    latexify(width_scale_factor=1, fig_height=1.5)  # width = 6 inch, height = 1.5 inch
    latexify(width_scale_factor=2, fig_height=2)  # width = 3 inch, height = 2 inch
    ```
    * Do not use `.pdf` or `.png` extension while saving a figure with `probml_utils.savefig`. For example:
    ```py
    savefig("foo")  # correct
    # savefig("foo.pdf")  # incorrect
    # savefig("foo.png")  # incorrect
    ```

4. To ensure your code passes the code formatting check, `pip install pre-commit` locally and run `pre-commit install`.
    * `pre-commit` will automatically format your notebook as per [this config](https://github.com/probml/pyprobml/blob/master/.pre-commit-config.yaml).
6. Follow PEP 8 naming convention.
7. Convert the existing code to `jax`, `flax` and `distrax` wherever possible.
8. (If applicable) In the last cell, create an interactive demo with `@interact`, `interactive` or any other tools available with `ipywidgets`. Make sure the demo works on Google colab.
9. Verify that your notebook does not throw any exception when executed in a fresh python/anaconda environment with the following command (you can make use of `docker` to have a fresh environment every time):

```bash
ipython foo.ipynb
```
10. At the time of opening a PR, double check that only the notebook you are working on is affected.

### Generating figures and saving them locally
* By default, figures are not saved:
```py
ipython foo.ipynb
```
* To save figures locally (in `.png` format):
```py
export FIG_DIR=/path/to/figures/directory
ipython foo.ipynb
```
* To save latexified figures locally (in `.pdf` format):
```py
export FIG_DIR=/path/to/figures/directory
export LATEXIFY=1
ipython foo.ipynb
```
* To save figures in both `.pdf` and `.png` formats:
```py
export FIG_DIR=/path/to/figures/directory
export DUAL_SAVE=1
ipython foo.ipynb
```

### Gotchas

* Use `latexify` function carefully with `seaborn`. Check if the generated figure is as expected.
* VS code does not behave well when using notebooks with `ipywidgets`, so double check with `jupyter notebook` GUI if something does not work as expected.

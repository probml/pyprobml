# Notebooks

## Instructions for contributors

### How to contribute?

1. Choose a figure from [book1](https://probml.github.io/pml-book/book1.html) ([pdf](https://github.com/probml/pml-book/releases/latest/download/book1.pdf)) or [book2](https://probml.github.io/pml-book/book2.html) ([pdf](https://github.com/probml/pml2-book/releases/latest/download/pml2.pdf)) and find out its source code notebook in this repo (mostly in the `notebooks` folder).
2. Fork the main repo and create a new branch for your figure.
3. Take [this notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book1/02/discrete_prob_dist_plot.ipynb) as a reference while converting the source code to a notebook.
4. To ensure your code passes the code formatting check, `pip install pre-commit` locally and run `pre-commit install`.
    * Install `black` for jupyter notebooks with: `pip install black[jupyter]` (bash) and `pip install 'black[jupyer]'` (zsh).
    * `pre-commit` will automatically format your notebook with `line-length=120` as set by [this config](https://github.com/probml/pyprobml/blob/master/.pre-commit-config.yaml).
6. Follow PEP 8 naming convention.
7. Convert the existing code to `jax`, `flax` and `distrax` wherever possible.
8. (If applicable) In the last cell, create an interactive demo with `@interact`, `interactive` or any other tools available with `ipywidgets`. Make sure the demo works on Google colab.
9. Please modify existing figure name by adding `_latexified` suffix e.g. `figures/uniform_distribution.pdf` -> `figures/uniform_distribution_latexified.pdf`.
10. Verify the notebook locally by running `pytest -s`.

### Generating figures and saving them locally
* To generate pdf figures, go to the root of the repo and use the following command:
```py
export FIG_DIR=/path/to/figures/directory
export LATEXIFY=1
ipython <notebook>.ipynb
```
For example:
```py
export FIG_DIR=figures
export LATEXIFY=1
ipython notebooks/book1/discrete_prob_dist_plot.ipynb
```

* By default, figures will not be saved anywhere. Upon setting `FIG_DIR`, they will be saved in `FIG_DIR` folder. Upon setting both `FIG_DIR` and `LATEXIFY`, they will be saved in `FIG_DIR` folder and latexified.

### Gotchas

* Use `latexify` function carefully with `seaborn`. Check if the generated figure is as expected.
* VS code does not behave well when using notebooks with `ipywidgets`, so double check with `jupyter notebook` GUI if something does not work as expected.

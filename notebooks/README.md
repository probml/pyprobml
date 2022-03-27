# Notebooks

## Instructions for contributors

### How to contribute?

1. Choose a figure from [book1](https://probml.github.io/pml-book/book1.html) or [book2](https://probml.github.io/pml-book/book2.html) and find out its source code in this repo (mostly in the `scripts` folder).
2. Fork the main repo and create a new branch for your figure.
3. Create a notebook with the same name as the source code file (e.g. `scripts/discrete_prob_dist_plot.py` -> `notebooks/discrete_prob_dist_plot.ipynb`).
4. Take [this notebook](https://github.com/probml/pyprobml/blob/master/notebooks/discrete_prob_dist_plot.ipynb) as a reference while converting the source code to a notebook.
5. To ensure your code passes the code formatting check, `pip install pre-commit` locally and run `pre-commit install`.
6. Follow PEP 8 naming convention.
7. Convert the existing code to `jax`, `flax` and `distrax` wherever possible.
8. (If applicable) In the last cell, create a demo with `@interact`, `interactive` or any other tools available with `ipywidgets`. Make sure the demo works on Google colab.
9. Please modify existing figure name by adding `_latexified` suffix e.g. `figures/uniform_distribution.pdf` -> `figures/uniform_distribution_latexified.pdf`.
10. Finally, add an entry for the new notebook in the `_toc.yml` file.

### Generating figures and saving them locally
* To generate pdf figures, go to the root of the repo and use the following command:
```py
DEV_MODE=1 ipython <notebook>.ipynb
```
For example:
```py
DEV_MODE=1 ipython notebooks/discrete_prob_dist_plot.ipynb
```

* Figures will be saved in the `figures` folder in the repo's root by default.
* Running the notebooks without the `DEV_MODE` flag will not save any figures and is equivalent to executing the notebooks normally (from GUI).

### Sharp edges

* Use `latexify` function carefully with `seaborn`. Check if the generated figure is as expected.
* VS code does not behave well when using notebooks with `ipywidgets`, so double check with `jupyter notebook` GUI if something does not work as expected.
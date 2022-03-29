This folder contains the auto-generated scripts from the notebooks present in the `notebooks` folder. Generation of these files takes place automatically via the GitHub workflow.

To run these scripts, please use `ipython` instead of `python` else `get_ipython()` will return `None`.

Example usage:
* Without saving the figures. No `latexify`.
```bash
ipython discrete_prob_dist_plot.py
```
* Save figures with `latexify`.
```bash
LATEXIFY=1 ipython discrete_prob_dist_plot.py
```

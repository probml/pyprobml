import os
from glob import glob
import pandas as pd
import nbformat
import argparse
import regex as re
import multiprocessing as mp

notebooks = glob("notebooks/book2/*/*.ipynb")  # + glob("notebooks/book1/*/*.ipynb")


def check_fun_to_notebook(notebook, fun):
    """
    fun should take one argument: code
    """
    try:
        nb = nbformat.read(notebook, as_version=4)
    except FileNotFoundError:
        print("File not found:", notebook)
        return False
    for cell in nb.cells:
        code = cell["source"]
        output = fun(code)
        if output:
            return True
    return False


def check_latexify(code):
    if "latexify" in code:
        return True
    return False


def execute_notebook(notebook):
    nb_name = notebook.split("/")[-1]
    print(f"Executing {notebook} ......")
    os.environ["LATEXIFY"] = ""
    os.environ["FIG_DIR"] = f"internal/figures/{nb_name}"
    os.system(f"ipython {notebook}")


latexified_nb = []
for no, nb in enumerate(notebooks):
    if check_fun_to_notebook(nb, check_latexify):
        latexified_nb.append(nb)


print(f"{len(latexified_nb)} notebooks are latexified")


n_cpu = mp.cpu_count() - 2
pool = mp.Pool(n_cpu)
pool.map(execute_notebook, latexified_nb)

from glob import glob
import sys

sys.path.append("misc")

from wrap_try_accept import apply_fun_to_notebook, remove_superimport, wrap_try_accept_in_code

book2_nb = glob("notebooks/book2/*/*.ipynb")

counter = 0
for nb in book2_nb:
    apply_fun_to_notebook(nb, remove_superimport)
    apply_fun_to_notebook(nb, wrap_try_accept_in_code)
    counter += 1

print("Done. {} notebooks are wrapped.".format(counter))

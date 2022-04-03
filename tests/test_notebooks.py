import pytest
import os
import subprocess
from glob import glob

os.environ["LATEXIFY"] = ""  # To enable latexify code
# os.environ["NO_SAVE_FIGS"] = ""  # To avoid saving the figures
notebooks1 = glob("notebooks/book1/*.ipynb")
notebooks2 = glob("notebooks/book2/*.ipynb")
notebooks = notebooks1 + notebooks2


@pytest.mark.parametrize("notebook", notebooks)
def test_run_notebooks(notebook):
    """
    Test notebooks
    """
    print("Testing notebook: {}".format(notebook))
    cmd = ["ipython", "-c", "%run {}".format(notebook)]
    subprocess.run(cmd, check=True)
    print("PASSED {}".format(notebook))

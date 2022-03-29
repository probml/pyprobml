import pytest
import os
import subprocess
from glob import glob

os.environ["LATEXIFY"] = ""  # To enable latexify code
os.environ["NO_SAVE_FIGS"] = ""  # To avoid saving the figures
notebooks = glob("notebooks/*.ipynb")


@pytest.mark.parametrize("notebook", notebooks)
def test_run_notebooks(notebook):
    """
    Test notebooks
    """
    print("Testing notebook: {}".format(notebook))
    cmd = ["ipython", "-c", "%run {}".format(notebook)]
    subprocess.run(cmd, check=True)
    print("PASSED {}".format(notebook))

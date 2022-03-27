import pytest
import os
import subprocess
from glob import glob

os.environ["DEV_MODE"] = "True"
os.environ["TEST_MODE"] = "True"
notebooks = glob("notebooks/*.ipynb")


@pytest.mark.parametrize("notebook", notebooks)
def test_run_notebooks(notebook):
    """
    Test notebooks
    """
    subprocess.run(["ipython", f"{notebook}"], check=True)

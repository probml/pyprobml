import pytest
import os
import subprocess
from glob import glob

os.environ["DEV_MODE"] = "True"
os.environ["TEST_MODE"] = "True"
scripts = glob("auto_generated_scripts/*.py")


@pytest.mark.parametrize("script", scripts)
def test_run_notebooks(script):
    """
    Test python scripts
    """
    subprocess.run(["ipython", f"{script}"], check=True)

import pytest
import os
import subprocess
from glob import glob

os.environ["DEV_MODE"] = "True"  # To enable latexify code
os.environ["TEST_MODE"] = "True"  # To avoid saving the figures
scripts = glob("auto_generated_scripts/*.py")


@pytest.mark.parametrize("script", scripts)
def test_run_notebooks(script):
    """
    Test python scripts
    """
    print("Testing script: {}".format(script))
    cmd = ["ipython", f"script"]
    subprocess.run(cmd, check=True)
    print("PASSED {}".format(script))

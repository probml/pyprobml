import sys

sys.path.append("internal/")
from wrap_try_accept import apply_fun_to_notebook
from glob import glob


def comment_pip_install(code):
    return code.replace("!pip install", "#!pip install")


def postprocessing(code):
    code = code.replace("%pip install ssm", "%pip install git+https://github.com/lindermanlab/ssm-jax-refactor.git")
    code = code.replace("%pip install google", "%pip install google-colab")
    code = code.replace("import pgmpy_utils as pgm", "import probml_utils.pgmpy_utils as pgm")
    code = code.replace("%pip install pgmpy_utils", "%pip install git+https://github.com/probml/probml-utils.git pgmpy")
    return code

def add_pip_qq_flag(code):
    if "pip install" in code:
        if "pip install -q" not in code:
            code = code.replace("pip install", "pip install -qq")
    return code
        

if __name__ == "__main__":
    notebooks = glob("notebooks/book1/*/*.ipynb") + glob("notebooks/book2/*/*.ipynb")
    for notebook in notebooks:
        print(f"******* {notebook} *******")
        apply_fun_to_notebook(notebook, add_pip_qq_flag)

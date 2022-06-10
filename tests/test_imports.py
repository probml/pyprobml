import pytest
from glob import glob
import pkgutil
import nbformat

# Global variables
TIMEOUT = 120

# Load notebooks
notebooks1 = glob("notebooks/book1/*/*.ipynb")
notebooks2 = glob("notebooks/book2/*/*.ipynb")
notebooks = notebooks1 + notebooks2

#get IGNORE_LIST of notebooks
IGNORE_LIST = []
with open("internal/ignored_notebooks.txt") as fp:
    ignored_notebooks = fp.readlines()
    for nb in ignored_notebooks:
        IGNORE_LIST.append(nb.strip().split("/")[-1])

def in_ignore_list(nb_path):
    nb_name = nb_path.split("/")[-1]
    return nb_name in IGNORE_LIST

notebooks = list(filter(lambda nb: not in_ignore_list(nb), notebooks))

# load installed modules
all_modules = set(map(lambda x: x[1], list(pkgutil.iter_modules())))

# Special cases
special_modules = set(["mpl_toolkits", "itertools", "time", "sys", "d2l", "augmax", "google"])
all_modules = all_modules.union(special_modules)


def get_simply_imported_module(line):
    line = line.rstrip()
    import_kw = None

    if line.startswith("import "):
        import_kw = "import "
    elif line.startswith("from ") and "import" in line:
        import_kw = "from "

    if import_kw:
        module = line[len(import_kw) :].split(" ")[0].split(".")[0]
        return module


def get_try_except_module(line):
    line = line.rstrip()
    import_kw = None

    if line.startswith(" ") and line.lstrip().startswith("import"):
        import_kw = "import "
    elif line.startswith(" ") and line.lstrip().startswith("from") and "import" in line:
        import_kw = "from "

    if import_kw:
        module = line.lstrip()[len(import_kw) :].split(" ")[0].split(".")[0]
        return module


# Parameterize notebooks
@pytest.mark.parametrize("notebook", notebooks)
def test_run_notebooks(notebook):
    """
    Test notebooks
    """
    nb = nbformat.read(notebook, as_version=4)
    lines = "\n".join(map(lambda x: x["source"], nb.cells)).split("\n")
    try_except_modules = set(map(get_try_except_module, lines))
    modules = set(filter(None, map(get_simply_imported_module, lines)))
    missing_modules = modules - all_modules - try_except_modules
    assert len(missing_modules) == 0, f"Missing {missing_modules} in {notebook}"


if __name__ == "__main__":
    for notebook in notebooks:
        test_run_notebooks(notebook)

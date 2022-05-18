import pytest
from glob import glob
import nbformat

# Load notebooks
notebooks1 = glob("notebooks/book1/*/*.ipynb")
notebooks2 = glob("notebooks/book2/*/*.ipynb")
notebooks = notebooks1 + notebooks2


def get_try_except_probml_utils_line(line):
    f"""
    check if import probml_utils is in given line: {line}
    if present, then return {line} wrapped with try...except
    """
    line = line.rstrip()
    if not line.startswith(" ") and "probml_utils" in line and "import" in line:
        try_except_line = (
            f"try:\n    {line}\nexcept ModuleNotFoundError:\n    %pip install git+https://github.com/probml/probml-utils.git\n    {line}"
        )
        return try_except_line
    return 0


# Parameterize notebooks
@pytest.mark.parametrize("notebook", notebooks)
def add_try_except_for_probml_util(notebook):
    nb = nbformat.read(notebook, as_version=4)
    for cell in nb.cells:
        lines = cell["source"].split("\n")
        for line_no, line in enumerate(lines):
            updated_line = get_try_except_probml_utils_line(line)
            if updated_line:
                lines[line_no] = updated_line
                break
        cell["source"] = "\n".join(lines)
        nbformat.write(nb, notebook)


if __name__ == "__main__":
    for notebook in notebooks:
        add_try_except_for_probml_util(notebook)

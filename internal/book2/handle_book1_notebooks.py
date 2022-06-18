from glob import glob
from itertools import count
import nbformat as nbf

book1_nb = glob("notebooks/book1/*/*.ipynb")
book1_nb_to_chap = {}
for nb in book1_nb:
    name = nb.split("/")[-1]
    chap = nb.split("/")[-2]
    book1_nb_to_chap[name] = chap

colab_base_url = "https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1"
prefix = "Source of this notebook is here:"

book2_nb = glob("notebooks/book2/*/*.ipynb")
counter = 0
for nb in book2_nb:
    name = nb.split("/")[-1]
    if name in book1_nb_to_chap:
        # read
        nb_content = nbf.read(nb, as_version=4)

        # replace with redirected link
        book1_chap = book1_nb_to_chap[name]

        # create new cell
        new_cell = nbf.v4.new_markdown_cell(f"{prefix} {colab_base_url}/{book1_chap}/{name}")
        nb_content["cells"] = [new_cell]

        # write
        nbf.write(nb_content, nb)

        print(f"{nb} is redirected to {colab_base_url}/{book1_chap}/{name}")
        counter += 1

print("Done. {} notebooks are redirected.".format(counter))

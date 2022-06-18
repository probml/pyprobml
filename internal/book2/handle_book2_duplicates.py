from glob import glob
import nbformat as nbf

book2_nb = glob("notebooks/book2/*/*.ipynb")

colab_base_url = "https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book2"
prefix = "Source of this notebook is here:"

# create notebook to chapters mapping
nb_to_chap = {}
for nb in book2_nb:
    name = nb.split("/")[-1]
    chap = nb.split("/")[-2]
    if name in nb_to_chap:
        nb_to_chap[name].append(chap)
    else:
        nb_to_chap[name] = [chap]

# keep first notebook and redirect others
counter = 0
for nb_name in nb_to_chap:
    chapters = sorted(nb_to_chap[nb_name])
    first_chap = chapters[0]
    for chapter in chapters[1:]:
        # read
        nb_content = nbf.read(f"notebooks/book2/{chapter}/{nb_name}", as_version=4)

        # replace with redirected link
        new_cell = nbf.v4.new_markdown_cell(f"{prefix} {colab_base_url}/{first_chap}/{nb_name}")
        nb_content["cells"] = [new_cell]

        # write
        nbf.write(nb_content, f"notebooks/book2/{chapter}/{nb_name}")
        print(f"{nb_name} duplicate in {chapter} is redirected to {first_chap}")
        counter += 1

print("Done. {} notebooks are redirected.".format(counter))

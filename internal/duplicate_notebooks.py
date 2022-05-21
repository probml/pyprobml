from glob import glob
import os
import shutil

book1_notebooks = glob("notebooks/book1/*/*.ipynb")
book2_notebooks = glob("notebooks/book2/*/*.ipynb")
misc_notebooks = glob("notebooks/misc/*.ipynb") + glob("notebooks/misc/*/*.ipynb")

print(len(book1_notebooks), len(book2_notebooks), len(misc_notebooks))

get_notebook_name = lambda notebook: notebook.split("/")[-1]

book1_notebooks_names = set(list(map(get_notebook_name, book1_notebooks)))
book2_notebooks_names = set(list(map(get_notebook_name, book2_notebooks)))
notebook_names = book1_notebooks_names.union(book2_notebooks_names)

for misc_notebook in misc_notebooks:
    notebook_name = get_notebook_name(misc_notebook)
    if notebook_name in notebook_names:
        print(f"{misc_notebook} is a duplicate")
        shutil.move(misc_notebook, f"deprecated/")
        # os.remove(misc_notebook)

    # handle _torch and _jax cases
    # notebook_name_t = notebook_name.replace(".ipynb","") + "_torch.ipynb"
    # if notebook_name_t in notebook_names:
    #     print(f"{notebook_name_t} is a duplicate")
    #     os.remove(misc_notebook)

    # notebook_name_j = notebook_name.replace(".ipynb","") + "_jax.ipynb"
    # if notebook_name_j in notebook_names:
    #     print(f"{notebook_name_j} is a duplicate")
    #     os.remove(misc_notebook)

# print(f"book1_notebooks_names: {len(book1_notebooks_names)} - {book1_notebooks_names}")
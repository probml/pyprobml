from glob import glob
import pandas as pd
import os
import nbformat
import numpy as np
import probml_utils.url_utils as url_utils

root_path = "notebooks"
curr_path = "internal/notebooks_md/"

# where to save notebooks.md
folder = "notebooks_md"
if os.path.exists(folder):
    os.system("rm -rf " + folder)
os.makedirs(folder)


def get_notebook_path(book_str, chap_no, nb_name):
    return os.path.join(root_path, book_str, chap_no, nb_name)


def split_nb_path(nb_path):
    """
    split nb path
    returns
    .........
    [book_no, chapter_no, notebook_name]
    """
    return nb_path.split("/")[-3:]


def is_query_in_nb(notebook, query):
    """
    fun should take one argument: code
    """
    nb = nbformat.read(notebook, as_version=4)
    for cell in nb.cells:
        code = cell["source"]
        if query in code:
            return 1
    return 0


def get_original_nb(df_nb_list_grp_ser):
    """
    from duplicate notebooks, identify notebook which does not
    have "Source of this notebook" keyword, which is original nb
    """
    nb_name = df_nb_list_grp_ser["Notebook"]
    books = df_nb_list_grp_ser["book_no"]
    chaps = df_nb_list_grp_ser["chap_no"]
    t = []
    for book, chap in zip(books, chaps):
        nb_path = get_notebook_path(book, chap, nb_name)
        is_source = is_query_in_nb(nb_path, "Source of this notebook")
        t.append(is_source)
    return t


def get_root_col(df_root_ser, col):
    f"""
    identify {col} having is_source = 0
    example:
           INPUT: df_root_ser[{col}] = [0,11,31], df_root_ser[is_source_present] = [1,0,1]
           OUTPUT: 11
    """
    is_source = df_root_ser["is_source_present"]
    nb_name = df_root_ser["Notebook"]

    if is_source.count(0) == 0:
        print(f"0: Not in pyprobml!:  {nb_name}")
        return df_root_ser[col][0]

    elif is_source.count(0) > 1:
        print(f"1: Multiple copies exist:  {nb_name}")

    else:
        return df_root_ser[col][is_source.index(0)]


def is_github_notebook(url):
    if url.startswith("https://github.com") and url.endswith(".ipynb"):
        return True
    return False


to_md_url = lambda text, url: f"[{text}]({url})"  # make md url


def to_colab_md_url(github_url):
    if is_github_notebook(github_url):
        colab_url = url_utils.github_url_to_colab_url(github_url)
        return to_md_url("colab", colab_url)
    else:
        return "NA"


# handle pyprobml's book1 & book2 notebooks
book1_notebooks = glob("notebooks/book1/*/*.ipynb") + glob("notebooks/book2/*/*.ipynb")
nb_list = list(map(split_nb_path, book1_notebooks))

# make dataframe of notebook
df_pyprobml = pd.DataFrame(nb_list, columns=["book_no", "chap_no", "Notebook"])

# group by duplicate notebooks
df_pyprobml = df_pyprobml.groupby("Notebook").agg(lambda x: list(x)).reset_index()

# check if "Source of this notebook" is present
df_pyprobml["is_source_present"] = df_pyprobml.apply(get_original_nb, axis=1)

# get root notebook
df_pyprobml["chap_no"] = df_pyprobml.apply(get_root_col, col="chap_no", axis=1)
df_pyprobml["book_no"] = df_pyprobml.apply(get_root_col, col="book_no", axis=1)

# Add github url
df_pyprobml["type"] = "github"
df_pyprobml["github_url"] = df_pyprobml.apply(
    lambda x: url_utils.make_url_from_chapter_no_and_script_name(
        chapter_no=int(x["chap_no"]),
        script_name=x["Notebook"],
        book_no=int(x["book_no"][-1]),
        convert_to_which_url="github",
    ),
    axis=1,
)

df_pyprobml = df_pyprobml[["Notebook", "type", "github_url"]]


# handle supplementary notebooks
supp_book = glob("notebooks/book2/*/*/*.ipynb") + glob("notebooks/book1/*/*/*.ipynb")
print(f"{len(supp_book)} supplementary notebooks found")

# convert to github url
github_root = "https://github.com/probml/pyprobml/blob/master/"
nb_github_colab_list = list(map(lambda x: [x.split("/")[-1], "github", github_root + x], supp_book))
df_supp = pd.DataFrame(nb_github_colab_list, columns=df_pyprobml.columns)


# handle external notebooks
df_external = pd.read_csv(curr_path + "external_links.csv")
common_notebooks = np.intersect1d(df_pyprobml["Notebook"].values, df_external["Notebook"].values)

# drop common notebooks from df_pyprobml and replace with df_external's notebooks
print(f"Before: {len(df_pyprobml)} pyprobml notebooks found")
df_pyprobml = df_pyprobml[~df_pyprobml["Notebook"].isin(common_notebooks)]
print(f"After: {len(df_pyprobml)} pyprobml notebooks found")
print(f"{len(df_external)} external reference found")


# combine all type of notebooks
df_all = pd.concat([df_pyprobml, df_supp, df_external])
df_all = df_all.sort_values(by="Notebook", key=lambda col: col.str.lower())

# get colab url from github url
df_all["colab_url"] = df_all["github_url"].apply(to_colab_md_url)

# enclose in span tag to give id as a Notebook
enclose_span = lambda text, nb_id: f"<span id={nb_id}>{text}</span>"
df_all["github_url"] = df_all.apply(
    lambda x: to_md_url(enclose_span(x["type"], x["Notebook"]), x["github_url"]), axis=1
)

# save to .md file
df_all = df_all[["Notebook", "github_url", "colab_url"]]
df_all.to_markdown(os.path.join(folder, "notebooks.md"), index=False)

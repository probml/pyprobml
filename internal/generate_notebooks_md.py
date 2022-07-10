from glob import glob
import pandas as pd
import os
import nbformat
import probml_utils.url_utils as url_utils

root_path = "notebooks"


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
    nb_name = df_nb_list_grp_ser["nb_name"]
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
    nb_name = df_root_ser["nb_name"]

    if is_source.count(0) == 0:
        print(f"0: Not in pyprobml!:  {nb_name}")
        return df_root_ser[col][0]

    elif is_source.count(0) > 1:
        print(f"1: Multiple copies exist:  {nb_name}")

    else:
        return df_root_ser[col][is_source.index(0)]


# handle book1
book1_notebooks = glob("notebooks/book1/*/*.ipynb") + glob("notebooks/book2/*/*.ipynb")
nb_list = list(map(split_nb_path, book1_notebooks))

# make dataframe of notebook
df_nb_list = pd.DataFrame(nb_list, columns=["book_no", "chap_no", "nb_name"])

# group by duplicate notebooks
df_nb_list_grp = df_nb_list.groupby("nb_name").agg(lambda x: list(x)).reset_index()

# check if "Source of this notebook" is present
df_nb_list_grp["is_source_present"] = df_nb_list_grp.apply(get_original_nb, axis=1)

# get root notebook
df_root = df_nb_list_grp
df_root["chap_no"] = df_nb_list_grp.apply(get_root_col, col="chap_no", axis=1)
df_root["book_no"] = df_nb_list_grp.apply(get_root_col, col="book_no", axis=1)


# Add colab url
df_root["colab_url"] = df_root.apply(
    lambda x: url_utils.make_url_from_chapter_no_and_script_name(
        chapter_no=int(x["chap_no"]),
        script_name=x["nb_name"],
        book_no=int(x["book_no"][-1]),
        convert_to_which_url="colab",
    ),
    axis=1,
)

# Add github url
df_root["github_url"] = df_root.apply(
    lambda x: url_utils.make_url_from_chapter_no_and_script_name(
        chapter_no=int(x["chap_no"]),
        script_name=x["nb_name"],
        book_no=int(x["book_no"][-1]),
        convert_to_which_url="github",
    ),
    axis=1,
)

enclose_span = lambda text, nb_id: f"<span id={nb_id}>{text}</span>"
to_md_url = lambda text, url: f"[{text}]({url})"

df_root["md_colab_url"] = df_root.apply(
    lambda x: to_md_url(enclose_span("colab", x["nb_name"]), x["colab_url"]), axis=1
)
df_root["md_github_url"] = df_root.apply(
    lambda x: to_md_url(enclose_span("github", x["nb_name"]), x["github_url"]), axis=1
)

df_final = df_root[["nb_name", "md_colab_url", "md_github_url"]]
df_final.columns = ["Notebook", "Colab url", "Github url"]


# handle supplementary notebooks
supp_book = glob("notebooks/book2/*/*/*.ipynb") + glob("notebooks/book1/*/*/*.ipynb")
print(f"{len(supp_book)} supplementary notebooks found")

# convert to github & colab url
github_root = "https://github.com/probml/pyprobml/blob/master/"
colab_root = "https://colab.research.google.com/github/probml/pyprobml/blob/master/"
nb_github_colab_list = list(map(lambda x: [x.split("/")[-1], colab_root + x, github_root + x], supp_book))

df_supp = pd.DataFrame(nb_github_colab_list, columns=df_final.columns)

# enclose id in <span>
df_supp["Colab url"] = df_supp.apply(lambda x: to_md_url(enclose_span("colab", x["Notebook"]), x["Colab url"]), axis=1)
df_supp["Github url"] = df_supp.apply(
    lambda x: to_md_url(enclose_span("github", x["Notebook"]), x["Github url"]), axis=1
)


# combine df_final and df_supp
df_chap_supp = pd.concat([df_final, df_supp])
df_chap_supp = df_chap_supp.sort_values(by="Notebook", key=lambda col: col.str.lower())

# save to .md file
df_chap_supp.to_markdown("notebooks.md", index=False)

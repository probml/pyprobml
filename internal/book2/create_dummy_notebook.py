"""
command usage:
python3 internal/book2/create_dummy_notebook.py --lof=internal/book2.lof --book_no=2
"""

import argparse
from email.policy import default
from random import choices
from TexSoup import TexSoup
import regex as re
import os
import nbformat as nbf
import pandas as pd
from glob import glob
from probml_utils.url_utils import (
    extract_scripts_name_from_caption,
    make_url_from_fig_no_and_script_name,
    figure_url_mapping_from_lof,
)

parser = argparse.ArgumentParser(description="create dummy notebook")
parser.add_argument("-lof", "--lof", type=str, help="")
parser.add_argument("-book_no", "--book_no", type=int, default=2, choices=[1, 2], help="")
parser.add_argument("-nb_path", "--nb_path", type=str, default="notebooks/", help="")

args = parser.parse_args()

lof_file = str(args.lof)
book_no = args.book_no
nb_path = args.nb_path


def convert_to_ipynb(file):
    if ".py" in file:
        return file[:-3] + ".ipynb"
    return file


def find_multinotebooks():
    fig_no_urls_mapping = figure_url_mapping_from_lof(lof_file, "", book_no=book_no)
    more_than_one = 0
    multi_notebooks = {}
    for fig_no in fig_no_urls_mapping:
        if "fig_" in fig_no_urls_mapping[fig_no]:
            print(fig_no_urls_mapping[fig_no])
            multi_notebooks[fig_no] = fig_no_urls_mapping[fig_no]
            more_than_one += 1
    print(f"{more_than_one} notebooks have more than one figure")
    return multi_notebooks


def delete_existing_multinotebooks():
    """
    delete existing notebooks
    """
    notebooks = glob(f"notebooks/book{book_no}/*/*.ipynb")
    cnt = 0
    for notebook in notebooks:
        if "fig_" in notebook.split("/")[-1]:
            os.remove(notebook)
            print(f"{notebook} deleted!")
            cnt += 1

    print(f"{cnt} notebooks deleted")


def preprocess_caption(captions):
    # create mapping of fig_no to list of script_name
    whole_link_ipynb = r"\{\S+\.ipynb\}"  # find {https://<path/to/>foo.ipynb}{foo.ipynb} from caption
    whole_link_py = r"\{\S+\.py\}"

    fig_cnt = 0
    cleaned_caption = {}

    multi_notebooks = find_multinotebooks()
    for caption in captions:
        fig_no = str(caption.contents[0])

        # if it does not contain multi_notebooks
        if fig_no not in multi_notebooks:
            continue

        caption = (
            str(caption)
            .replace(r"\ignorespaces", "")
            .replace(r" \relax", "")
            .replace(r"\href", "")
            .replace(r"\url", "")
            .replace(r'\cc@accent {"705E}', "")
            .replace(r"\numberline", "")
            .replace(r"\bm", "")
            .replace(r"\DOTSB", "")
            .replace(r"\slimits", "")
            .replace(r"\oset", "")
        )

        # print(fig_no, end=" ")
        links = re.findall(whole_link_ipynb, str(caption)) + re.findall(whole_link_py, str(caption))
        # print(fig_no, links)
        for link in links:
            script = extract_scripts_name_from_caption(link)[0]
            script_ipynb = convert_to_ipynb(script)
            original_url = f"[{script_ipynb}]({make_url_from_fig_no_and_script_name(fig_no,script_ipynb, book_no = book_no)})"  # in form of markdown hyperlink
            caption = caption.replace(link, original_url)

        caption = re.findall(r"{\d+.\d+}{(.*)}", caption)[0].strip()  # extract caption from {4.13}{caption}

        # print(fig_no, caption, end="\n\n")
        cleaned_caption[fig_no] = caption

    return cleaned_caption


def parse_lof(lof_file):
    with open(lof_file) as fp:
        LoF_File_Contents = fp.read()
    return LoF_File_Contents


def make_dummy_notebook_name(fig_no):
    """
    convert 1.11 to fig_1_11.ipynb
    """
    return f"fig_{fig_no.replace('.','_')}.ipynb"


def create_multi_notebooks(cleaned_captions, relative_path=nb_path):
    """
    create new notebook and add caption to it
    """
    # https://stackoverflow.com/questions/38193878/how-to-create-modify-a-jupyter-notebook-from-code-python
    cnt = 0
    for fig_no in cleaned_captions:

        # make relative path for new dummy notebook
        chapter_no = int(fig_no.split(".")[0])

        dummpy_notebook = make_dummy_notebook_name(fig_no)
        fig_path = os.path.join(relative_path, f"book{book_no}/{chapter_no:02d}", dummpy_notebook)
        print(fig_path.split("/")[-1], end="\n")

        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_markdown_cell(cleaned_captions[fig_no])]
        with open(fig_path, "w") as f:
            nbf.write(nb, f)
            cnt += 1

    print(f"\n{cnt} notebooks written!")


if __name__ == "__main__":
    # delete existing multinotebooks
    delete_existing_multinotebooks()

    # find multinotebooks
    print(find_multinotebooks())

    # parse lof file
    soup = TexSoup(parse_lof(lof_file))

    # preprocess caption
    cleaned_captions = preprocess_caption(soup.find_all("numberline"))

    # create multinoteboos and write caption
    create_multi_notebooks(cleaned_captions)

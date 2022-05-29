from typing import Any
from TexSoup import TexSoup
import regex as re
import os
import pandas as pd
import os
from glob import glob
import nbformat
from probml_utils.url_utils import extract_scripts_name_from_caption, make_url_from_fig_no_and_script_name, dict_to_csv
import argparse


def hyperlink_from_urls(urls):
    """
    convert url to [nb_name](url)
    """
    hyperlinks = ""
    for url in urls:
        nb_name = url.split("/")[-1]
        hyperlinks += f"[{nb_name}]({url})"


def figure_url_mapping_from_lof_dummy_nb_excluded(
    lof_file_path,
    csv_name,
    convert_to_colab_url=True,
    base_url="https://github.com/probml/pyprobml/blob/master/notebooks",
    book_no=1,
):
    f"""
    create mapping of fig_no to url by parsing lof_file and save mapping in {csv_name}
    """
    with open(lof_file_path) as fp:
        LoF_File_Contents = fp.read()
    soup = TexSoup(LoF_File_Contents)

    # create mapping of fig_no to list of script_name

    url_mapping = {}
    for caption in soup.find_all("numberline"):
        fig_no = str(caption.contents[0])
        extracted_scripts = extract_scripts_name_from_caption(str(caption))
        if len(extracted_scripts) > 0:
            url_mapping[fig_no] = []
            for script_name in extracted_scripts:
                url_mapping[fig_no].append(
                    make_url_from_fig_no_and_script_name(
                        fig_no,
                        script_name,
                        convert_to_colab_url=convert_to_colab_url,
                        base_url=base_url,
                        book_no=book_no,
                    )
                )

    if csv_name:
        dict_to_csv(url_mapping, csv_name)
    print(f"Mapping of {len(url_mapping)} urls is saved in {csv_name}")
    return url_mapping


def get_chap_mapping(csv_name):
    """
    return mapping of chap_no to chap_name
    """
    df = pd.read_csv(csv_name, dtype=str)
    return dict(zip(df["chap_no"], df["chap_name"]))


def chap_to_urls_mapping(csv_name):
    """
    return like,
    {
        "1": {
            "1.1":[url1,url2],
            "1.3 :[url1]
        }
        "2" : {
            "2.2":[url1,url2],
            "2.4":[url1]
        }
    }
    """
    df = pd.read_csv(csv_name, dtype=str)
    chap_nb = {}

    def chap_nb_wise_mapping(x):
        chap_no = x["key"].split(".")[0]
        try:
            chap_nb[chap_no][x["key"]] = eval(x["url"])
        except:
            chap_nb[chap_no] = {}
            chap_nb[chap_no][x["key"]] = eval(x["url"])

    df.apply(chap_nb_wise_mapping, axis=1)
    return chap_nb


def save_to_md(content, md_name):
    with open(md_name, "w") as fp:
        fp.write(content)


def check_latexify(code):
    if "latexify" in code:
        return True
    return False


def check_jaxify(code):
    if "import jax" in code or "import flax" in code or "from jax" in code or "from flax" in code:
        return True
    return False


def check_fun_to_notebook(notebook, fun):
    """
    fun should take one argument: code
    """
    nb = nbformat.read(notebook, as_version=4)
    for cell in nb.cells:
        code = cell["source"]
        output = fun(code)
        if output:
            return True
    return False


def get_path_from_url(url, base_path="notebooks"):
    """
    extract book_no/chap_name/nb_name.ipynb from url
    """
    book_no, chap_no, nb_name = url.split("/")[-3], url.split("/")[-2], url.split("/")[-1]
    path = os.path.join(base_path, book_no, chap_no, nb_name)
    return path


parser = argparse.ArgumentParser(description="create figure dashboard")
parser.add_argument("-user_name", "--user_name", type=str, help="", default="probml/pyprobml")
args = parser.parse_args()
user, repo = args.user_name.split("/")  # github.repository gives owner/repo

repo_root = f"https://github.com/{user}/{repo}"
run_root = f"https://raw.githubusercontent.com/{user}/{repo}/workflow_testing_indicator"
get_run = (
    lambda book_no, chapter_num, nb_name: f'<img width="20" alt="image" src={run_root}/notebooks/book{book_no}/{int(chapter_num):02d}/{nb_name.replace(".ipynb", ".png")}>, [log]({run_root}/notebooks/book{book_no}/{int(chapter_num):02d}/{nb_name.replace(".ipynb", ".log")})'
)
right_emoji = "&#9989;"
wrong_emoji = "&#10060;"
book_no = 1
lof_path = f"internal/book{book_no}.lof"
csv_excluded_dummy = f"internal/figures_url_mapping_book{book_no}_excluded_dummy_nb.csv"

# figure_url_mapping_from_lof_dummy_nb_excluded(lof_path, csv_excluded_dummy, book_no=1)
chap_urls_mapping = chap_to_urls_mapping(csv_excluded_dummy)

csv_chap_names = f"internal/chapter_no_to_name_mapping_book{book_no}.csv"
chap_no_chap_name_mapping = get_chap_mapping(csv_chap_names)
# print(chap_urls_mapping)
md_content = ""
for chap_no in chap_urls_mapping:
    notebooks = []
    base_str = f"<details>\n<summary>Chapter: {chap_no}_{chap_no_chap_name_mapping[chap_no]}</summary>\n\n"
    urls_list = chap_urls_mapping[chap_no]
    print(f"*** {chap_no}-{chap_no_chap_name_mapping[chap_no]}: {len(urls_list)} urls found! ****")
    for fig_no in urls_list:
        # print(fig_no, type(urls_list[fig_no]), urls_list[fig_no])
        for url in urls_list[fig_no]:
            # print(url)
            path = get_path_from_url(url)
            # print(path)
            is_latexify = int(check_fun_to_notebook(path, check_latexify))
            is_jaxify = int(check_fun_to_notebook(path, check_jaxify))
            notebooks.append(
                {
                    "nb_name": f"[{url.split('/')[-1]}]({url})",
                    "fig_no": fig_no,
                    "workflow": get_run(book_no, chap_no, url.split("/")[-1]),
                    "latexify": right_emoji if is_latexify else wrong_emoji,
                    "jaxify": right_emoji if is_jaxify else wrong_emoji,
                }
            )

    if len(notebooks) > 0:
        df_nb = pd.DataFrame(notebooks)
        md = base_str + df_nb.to_markdown(index=None) + "\n</details>"
        # print(md)
        md_content += md + "\n\n"

save_to_md(md_content, f"workflow_testing_indicator/dashboard_figures_book{book_no}.md")

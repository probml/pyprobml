"""
command usage:
python3 internal/book1/figure_no_notebook_mapping.py -lof internal/book1.lof -csv internal/book1/figures_nb_mapping_book1.csv
"""
import argparse
from probml_utils.url_utils import (
    dict_to_csv,
    extract_scripts_name_from_caption,
)
from TexSoup import TexSoup

parser = argparse.ArgumentParser(description="create figure url mapping")
parser.add_argument("-lof", "--lof", type=str, help="", default="internal/book2.lof")
parser.add_argument("-csv", "--csv", type=str, help="", default="internal/figures_url_mapping_book2.csv")
args = parser.parse_args()

BOOK_NO = int(args.lof.split("/")[-1].split(".")[0][-1])  # from internal/book1.lof to 1
NOTEBOOKS_MD_URL = "https://probml.github.io/notebooks#"

def fig_no_nb_mapping(lof_file_path, csv_name, make_url = False):
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
        nb = None
        if len(extracted_scripts) == 1:
            nb = extracted_scripts[0]

        elif len(extracted_scripts) > 1: # use dummy notebooks
            chap, fig = fig_no.split(".")
            nb = f"fig_{chap}_{fig}.ipynb"
        
        if nb:
            if make_url: 
                url_mapping[fig_no] = NOTEBOOKS_MD_URL + nb
            else:
                url_mapping[fig_no] = nb


    if csv_name:
        dict_to_csv(url_mapping, csv_name)
    print(f"Mapping of {len(url_mapping)} urls is saved in {csv_name}")
    return url_mapping


print(f"Parsing started from {args.lof}...........")

if args.csv:
    figure_mapping = fig_no_nb_mapping(args.lof, args.csv)
    figure_mapping = fig_no_nb_mapping(args.lof, args.csv.replace(".csv","")+"_urls.csv", make_url=True)

else:
    figure_mapping = fig_no_nb_mapping(args.lof, "")
    figure_mapping = fig_no_nb_mapping(args.lof, args.csv.replace("csv","")+"_urls.csv", make_url=True)


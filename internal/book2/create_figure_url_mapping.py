"""
command usage:
python3 internal/book2/create_figure_url_mapping.py -lof internal/book2.lof -csv internal/figures_url_mapping_book2.csv
"""
import argparse
import regex as re
from probml_utils.url_utils import (
    figure_url_mapping_from_lof,
    dict_to_csv,
    make_url_from_fig_no_and_script_name,
    extract_scripts_name_from_caption,
)
from TexSoup import TexSoup

parser = argparse.ArgumentParser(description="create figure url mapping")
parser.add_argument("-lof", "--lof", type=str, help="", default="internal/book2.lof")
parser.add_argument("-csv", "--csv", type=str, help="", default="internal/figures_url_mapping_book2.csv")
args = parser.parse_args()

BOOK_NO = int(args.lof.split("/")[-1].split(".")[0][-1])  # from internal/book1.lof to 1


def figure_url_mapping_from_lof_dummy_nb_excluded(
    lof_file_path,
    csv_name,
    convert_to_which_url="colab",
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
                        convert_to_which_url=convert_to_which_url,
                        base_url=base_url,
                        book_no=book_no,
                    )
                )

    if csv_name:
        dict_to_csv(url_mapping, csv_name)
    print(f"Mapping of {len(url_mapping)} urls is saved in {csv_name}")
    return url_mapping


print(f"Parsing started from {args.lof}...........")

if args.csv:
    figure_mapping = figure_url_mapping_from_lof(args.lof, args.csv, book_no=BOOK_NO)
    figure_mapping = figure_url_mapping_from_lof_dummy_nb_excluded(
        args.lof, args.csv.replace(".csv", "") + "_excluded_dummy_nb.csv", book_no=BOOK_NO
    )
else:
    figure_mapping = figure_url_mapping_from_lof(args.lof, "", book_no=BOOK_NO)
    figure_mapping = figure_url_mapping_from_lof_dummy_nb_excluded(args.lof, "", book_no=BOOK_NO)

import argparse
from probml_utils.url_utils import non_figure_notebook_url_mapping
from glob import glob

parser = argparse.ArgumentParser(description="create non figure url mapping")
parser.add_argument("-book_no", "--book_no", type=int ,default=2, help="")
parser.add_argument("-csv", "--csv" ,type=str, help="")
args = parser.parse_args()

book_no = args.book_no
nb_path = f"notebooks/book{book_no}/*/*.ipynb"
notebooks_1 = glob(nb_path)
print(notebooks_1)

print(f"Parsing started from {nb_path}...........")

if args.csv:
    figure_mapping = non_figure_notebook_url_mapping(notebooks_1, args.csv, book_no=2)
else:
    figure_mapping = non_figure_notebook_url_mapping(notebooks_1, "", book_no=2)
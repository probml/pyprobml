import os 
from glob import glob
from probml_utils.url_utils import (
    dict_to_csv
)

notebooks = glob("notebooks/book1/*/*.ipynb") + glob("notebooks/book1/*/*/*.ipynb")
NOTEBOOKS_MD_URL = "https://probml.github.io/notebooks#"


mapping = {}
for nb in notebooks:
    nb_name = nb.split("/")[-1]
    mapping[nb_name] = NOTEBOOKS_MD_URL + nb_name 

dict_to_csv(mapping, "internal/book1/non_figures_nb_mapping_book1_urls.csv")
print(f"{len(notebooks)} notebooks saved")
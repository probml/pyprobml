import os
from glob import glob
import pandas as pd
import nbformat

bookv2_path = "../../bookv2"

latexified_figs = glob("internal/figures/*/*_latexified.pdf")
latexified_figs = set(map(lambda x: x.split("/")[-1], latexified_figs))

bookv2_figs = glob(os.path.join(bookv2_path, "figures/*_latexified.pdf"))
bookv2_figs = set(map(lambda x: x.split("/")[-1], bookv2_figs))


print(len(latexified_figs), len(bookv2_figs))

print("Left book2 figures")
print(latexified_figs - bookv2_figs)

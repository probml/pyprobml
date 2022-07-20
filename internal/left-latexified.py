import os
from glob import glob


bookv2_path = "../../bookv2"


def to_latex_nb_name(notebook):
    return notebook.replace("_", "\_").replace(".ipynb", "")


latexified_figs = glob("internal/figures/*/*_latexified.pdf")
# latexified_figs = set(map(lambda x: x.split("/")[-1], latexified_figs))

bookv2_figs = glob(os.path.join(bookv2_path, "figures/*_latexified.pdf"))
bookv2_figs = set(map(lambda x: x.split("/")[-1], bookv2_figs))

print(len(latexified_figs), len(bookv2_figs))
ignored_nb = [
    "gp\_deep\_kernel\_learning",
    "simulated\_annealing\_2d\_demo",
    "gp\_kernel\_opt",
    "linreg\_height\_weight",
]
for latexified_nb_path in latexified_figs:
    latexified_fig = latexified_nb_path.split("/")[-1]
    latexified_nb = to_latex_nb_name(latexified_nb_path.split("/")[-2])
    if latexified_fig not in bookv2_figs and latexified_nb not in ignored_nb:
        print(f"{latexified_nb} \t {latexified_fig}")

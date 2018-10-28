# figures for structured video models

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

import daft
import os

pgm = daft.PGM([8, 9], origin=[0, 0], observed_style="inner")

x = 1
pgm.add_node(daft.Node("It1", r"$I_{t-1}$", x, 1))
pgm.add_node(daft.Node("ft1", r"f", x, 2))
pgm.add_node(daft.Node("bt1", r"$b_{t-1}$", x, 3))
pgm.add_node(daft.Node("LSt1", r"$L_{t-1}^S$", x, 4, observed=True))
pgm.add_node(daft.Node("St1", r"$S_{t-1}$", x, 5, observed=True))


pgm.add_edge("It1", "ft1")
pgm.add_edge("ft1", "bt1")
pgm.add_edge("bt1", "LSt1")
pgm.add_edge("St1", "LSt1")



x = 3
pgm.add_node(daft.Node("It", r"$I_{t}$", x, 1))
pgm.add_node(daft.Node("ft", r"f", x, 2))
pgm.add_node(daft.Node("bt", r"$b_{t}$", x, 3))
pgm.add_node(daft.Node("LSt", r"$L_{t}^S$", x, 4, observed=True))
pgm.add_node(daft.Node("St", r"$S_{t}$", x, 5, observed=True))


pgm.add_edge("It", "ft")
pgm.add_edge("ft", "bt")
pgm.add_edge("bt", "LSt")
pgm.add_edge("St", "LSt")

pgm.add_edge("bt1", "ft", linestyle=":")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "video-vae"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())

import os

import imp
#daft = imp.load_source('daft', 'daft-080308/daft.py')
import daft

# DRN

pgm = daft.PGM([4, 6], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h$", 1, 1))
pgm.add_node(daft.Node("h2", r"$h$", 1, 2))
pgm.add_node(daft.Node("h3", r"$h_i^l$", 1, 3))
pgm.add_node(daft.Node("h4", r"$h$", 1, 4))
pgm.add_node(daft.Node("h5", r"$h$", 1, 5))

pgm.add_node(daft.Node("hh1", r"$h_{i1}^l$", 2, 4))
pgm.add_node(daft.Node("hh2", r"$h_{ik}^l$", 2, 2))

pgm.add_node(daft.Node("hhh", r"$h_i^{l+1}$", 3, 3))


pgm.add_edge("h1", "hh1", linestyle=":")
pgm.add_edge("h2", "hh1", linestyle=":")
pgm.add_edge("h3", "hh1", linestyle=":")
pgm.add_edge("h4", "hh1", linestyle="-")
pgm.add_edge("h5", "hh1", linestyle=":")

pgm.add_edge("h1", "hh2", linestyle=":")
pgm.add_edge("h2", "hh2", linestyle="-")
pgm.add_edge("h3", "hh2", linestyle=":")
pgm.add_edge("h4", "hh2", linestyle=":")
pgm.add_edge("h5", "hh2", linestyle=":")

pgm.add_edge("hh1", "hhh", linestyle="-")
pgm.add_edge("hh2", "hhh", linestyle="-")

pgm.add_edge("h3", "hhh", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "DRN"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))




# RN

pgm = daft.PGM([4, 6], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h$", 1, 1))
pgm.add_node(daft.Node("h2", r"$h$", 1, 2))
pgm.add_node(daft.Node("h3", r"$h_i^l$", 1, 3))
pgm.add_node(daft.Node("h4", r"$h$", 1, 4))
pgm.add_node(daft.Node("h5", r"$h$", 1, 5))

pgm.add_node(daft.Node("hhh", r"$h_i^{l+1}$", 3, 3))


pgm.add_edge("h1", "hhh", linestyle="-")
pgm.add_edge("h2", "hhh", linestyle="-")
pgm.add_edge("h3", "hhh", linestyle="-")
pgm.add_edge("h4", "hhh", linestyle="-")
pgm.add_edge("h5", "hhh", linestyle="-")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "RN"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))



# AN

pgm = daft.PGM([4, 6], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h$", 1, 1))
pgm.add_node(daft.Node("h2", r"$h$", 1, 2))
pgm.add_node(daft.Node("h3", r"$h_i^l$", 1, 3))
pgm.add_node(daft.Node("h4", r"$h$", 1, 4))
pgm.add_node(daft.Node("h5", r"$h$", 1, 5))

pgm.add_node(daft.Node("hhh", r"$h_i^{l+1}$", 3, 3))


pgm.add_edge("h1", "hhh", linestyle=":")
pgm.add_edge("h2", "hhh", linestyle=":")
pgm.add_edge("h3", "hhh", linestyle=":")
pgm.add_edge("h4", "hhh", linestyle="-")
pgm.add_edge("h5", "hhh", linestyle=":")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "AN"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))
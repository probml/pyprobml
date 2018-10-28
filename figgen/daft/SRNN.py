from matplotlib import rc
import matplotlib.pyplot as plt
rc("font", family="serif", size=12)
rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())
#rc('text.latex',   preamble="\usepackage{amssymb}\usepackage{amsmath}\usepackage{mathrsfs}")
   
import os

import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')
#import daft
folder = "/Users/kpmurphy/github/pyprobml/figures"



pgm = daft.PGM([3, 3], origin=[0, 0])

pgm.add_node(daft.Node("x1", r"$x_{t-1}$", 1, 1))
pgm.add_node(daft.Node("z1", r"$z_{t-1}$", 1, 2))
pgm.add_node(daft.Node("x2", r"$x_{t}$", 2, 1))
pgm.add_node(daft.Node("z2", r"$z_{t}$", 2, 2))

pgm.add_edge("z1", "z2", linestyle="-")
pgm.add_edge("z1", "x1", linestyle="-")
pgm.add_edge("z2", "x2", linestyle="-")

pgm.render()
fname = "SSM-simple"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))




pgm = daft.PGM([3, 3], origin=[0, 0])

pgm.add_node(daft.Node("x1", r"$x_{t-1}$", 1, 1))
pgm.add_node(daft.Node("z1", r"$h_{t-1}$", 1, 2.0,  shape="rectangle"))
pgm.add_node(daft.Node("x2", r"$x_{t}$", 2, 1))
pgm.add_node(daft.Node("z2", r"$h_{t}$", 2, 2.0,  shape="rectangle"))

pgm.add_edge("z1", "z2", linestyle="-")
pgm.add_edge("z1", "x1", linestyle="-")
pgm.add_edge("z2", "x2", linestyle="-")
pgm.add_edge("x1", "z2", linestyle="-")
pgm.add_edge("x1", "x2", linestyle=":")

pgm.render()
fname = "RNN-simple"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))



pgm = daft.PGM([3, 4], origin=[0, 0])

pgm.add_node(daft.Node("x1", r"$x_{t-1}$", 1, 1))
pgm.add_node(daft.Node("z1", r"$z_{t-1}$", 1, 3))
pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 1, 2.0,  shape="rectangle"))
pgm.add_node(daft.Node("x2", r"$x_{t}$", 2, 1))
pgm.add_node(daft.Node("z2", r"$z_{t}$", 2, 3))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 2, 2.0,  shape="rectangle"))

pgm.add_edge("z1", "h1", linestyle="-")
pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h1", "x1", linestyle="-")
pgm.add_edge("z2", "h2", linestyle="-")
pgm.add_edge("h2", "x2", linestyle="-")
pgm.add_edge("x1", "h2", linestyle="-")
pgm.add_edge("x1", "x2", linestyle=":")

pgm.render()
fname = "SRNN-simple"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))
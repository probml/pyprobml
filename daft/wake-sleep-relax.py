
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())

import os

import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')
#import daft

pgm = daft.PGM([6, 6], origin=[0, 0])

pgm.add_node(daft.Node("x", r"$x$", 4, 1, observed=True))
pgm.add_node(daft.Node("z", r"$z$", 3, 4))
pgm.add_node(daft.Node("eps", r"$\epsilon$", 5, 4))
pgm.add_node(daft.Node("psi", r"$\theta_z$", 3, 5))
pgm.add_node(daft.Node("theta", r"$\theta_e$", 5, 5))

pgm.add_node(daft.Node("fInf", r"Inf", 2.0, 3,  shape="rectangle"))
pgm.add_node(daft.Node("fRender", r"R", 4.0, 2,  shape="rectangle"))

pgm.add_edge("psi", "z")
pgm.add_edge("theta", "eps")
pgm.add_edge("z", "fRender")
pgm.add_edge("eps", "fRender")
pgm.add_edge("fRender", "x")

pgm.add_edge("x", "fInf")
pgm.add_edge("fInf", "z")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "wake-sleep-relax"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
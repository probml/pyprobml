
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())

import os

import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')
import daft

# Structured Filter and predict 

pgm = daft.PGM([5, 4], origin=[0, 4])

pgm.add_node(daft.Node("s0", r"$s_{t-1}$", 1, 6))
pgm.add_node(daft.Node("p0", r"$p_{t-1}$", 1, 5))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 2, 7))
pgm.add_node(daft.Node("FSU1", r"$F_{su}$", 2.0, 6, shape="rectangle"))
pgm.add_node(daft.Node("FPU1", r"$F_{pu}$", 2.0, 5, shape="rectangle"))
pgm.add_node(daft.Node("s1", r"$s_{t}$", 3, 6))
pgm.add_node(daft.Node("p1", r"$p_{t}$", 3, 5))


pgm.add_edge("s0", "FSU1", linestyle="-")
pgm.add_edge("p0", "FPU1", linestyle="-")
pgm.add_edge("x1", "FSU1", linestyle="-")
pgm.add_edge("FSU1", "s1", linestyle="-")
pgm.add_edge("FPU1", "p1", linestyle="-")
pgm.add_edge("s1", "FPU1", linestyle="-")


pgm.add_edge("s1", "FPU1", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-FG-struct-filter"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))



# Structured Filter and predict 

pgm = daft.PGM([5, 8], origin=[0, 0])

pgm.add_node(daft.Node("s0", r"$s_{t-1}$", 1, 6))
pgm.add_node(daft.Node("p0", r"$p_{t-1}$", 1, 5))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 2, 7))
pgm.add_node(daft.Node("FSU1", r"$F_{su}$", 2.0, 6, shape="rectangle"))
pgm.add_node(daft.Node("FPU1", r"$F_{pu}$", 2.0, 5, shape="rectangle"))
pgm.add_node(daft.Node("FSP1", r"$F_{sp}$", 2.0, 4, shape="rectangle"))
pgm.add_node(daft.Node("FPP1", r"$F_{pp}$", 2.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("s1", r"$s_{t}$", 3, 6))
pgm.add_node(daft.Node("p1", r"$p_{t}$", 3, 5))
pgm.add_node(daft.Node("st1", r"$\tilde{s}_{t}$", 4, 4))
pgm.add_node(daft.Node("pt1", r"$\tilde{p}_{t}$", 4, 3))
pgm.add_node(daft.Node("FD1", r"$I$", 3.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("FDt1", r"$I$", 4.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("po1", r"$p_{t}$", 3, 1))
pgm.add_node(daft.Node("pot1", r"$\tilde{p}_{t}$", 4, 1))

pgm.add_edge("s0", "FSU1", linestyle="-")
pgm.add_edge("p0", "FPU1", linestyle="-")
pgm.add_edge("x1", "FSU1", linestyle="-")
pgm.add_edge("FSU1", "s1", linestyle="-")
pgm.add_edge("FPU1", "p1", linestyle="-")
pgm.add_edge("s0", "FSP1", linestyle="-")
pgm.add_edge("p0", "FPP1", linestyle="-")

pgm.add_edge("FSP1", "st1", linestyle="-")
pgm.add_edge("FPP1", "pt1", linestyle="-")
pgm.add_edge("p1", "FD1", linestyle="-")
pgm.add_edge("pt1", "FDt1", linestyle="-")
pgm.add_edge("FD1", "po1", linestyle="-")
pgm.add_edge("FDt1", "pot1", linestyle="-")

pgm.add_edge("s1", "FPU1", linestyle="-")
pgm.add_edge("st1", "FPP1", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-FG-struct-filter-predict"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
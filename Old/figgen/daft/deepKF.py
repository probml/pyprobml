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



pgm = daft.PGM([4, 7], origin=[0, -1])

pgm.add_node(daft.Node("Gold", r"$G^z_{t-1}$", 1, 3))
pgm.add_node(daft.Node("Gpred", r"$\hat{G}^{z}_{t}$", 2, 3))
pgm.add_node(daft.Node("Gnew", r"$G^z_{t}$", 3, 3))

pgm.add_node(daft.Node("alpha", r"$\alpha_{t}$", 1, 5))
pgm.add_node(daft.Node("A", r"$A_{t}$", 2, 4))
pgm.add_node(daft.Node("C", r"$C_{t}$", 3, 4))

pgm.add_node(daft.Node("priorh", r"$\hat{G}^x_{t}$", 2, 2))
pgm.add_node(daft.Node("posth", r"$G^x_{t}$", 3, 2))
pgm.add_node(daft.Node("px", r"$\hat{p}^y_{t}$", 2, 1))
pgm.add_node(daft.Node("L", r"$L_{t}$", 2, 0))

pgm.add_node(daft.Node("x", r"$o_{t}$", 3, 1))
pgm.add_node(daft.Node("y", r"$y_{t}$", 3, 0))

pgm.add_edge("Gold", "Gpred", linestyle="-")
pgm.add_edge("Gpred", "Gnew", linestyle="-")
pgm.add_edge("Gold", "alpha", linestyle="-")
pgm.add_edge("alpha", "A", linestyle="-")
pgm.add_edge("alpha", "C", linestyle="-")

pgm.add_edge("A", "Gpred", linestyle="-")
pgm.add_edge("C", "Gnew", linestyle="-")
pgm.add_edge("Gpred", "priorh", linestyle="-")
pgm.add_edge("priorh", "px", linestyle="-")
pgm.add_edge("px", "L", linestyle="-")
pgm.add_edge("y", "L", linestyle="-")
pgm.add_edge("x", "posth", linestyle="-")
pgm.add_edge("posth", "Gnew", linestyle="-")




pgm.render()
fname = "deepKF"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))
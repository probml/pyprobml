# pose model

#from matplotlib import rc
#rc("font", family="serif", size=12)
#rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())

#import daft
import os

import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')

pgm = daft.PGM([4, 4], origin=[0, 0], observed_style="inner")

pgm.add_node(daft.Node("k", r"$k$", 1, 1))
pgm.add_node(daft.Node("x", r"$x$", 2, 2))
pgm.add_node(daft.Node("kk", r"$k'$", 3, 2))

pgm.add_edge("k", "x")
pgm.add_edge("x", "kk")
pgm.add_edge("k", "kk")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "pose-eccv18"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))

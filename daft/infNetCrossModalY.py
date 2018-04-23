from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

import daft
import os

pgm = daft.PGM([10, 5], origin=[0, 0])

pgm.add_node(daft.Node("xnm", r"$x_{nm}$", 1, 1))
pgm.add_node(daft.Node("hnm1", r"$h_{nm}^1$", 1, 2))
pgm.add_node(daft.Node("hnm2", r"$h_{nm}^{L-1}$", 1, 3))

pgm.add_node(daft.Node("znm1", r"$z_{nm}^1$", 2, 2))
pgm.add_node(daft.Node("znm2", r"$z_{nm}^{L-1}$", 2, 3))

pgm.add_node(daft.Node("Lnmx", r"$\calL_{nm}^x$", 3, 1))
pgm.add_node(daft.Node("Lnm1", r"$\calL_{nm}^1$", 3, 2))
pgm.add_node(daft.Node("Lnm2", r"$\calL_{nm}^L$", 3, 3))

pgm.add_node(daft.Node("xsm", r"$x_{*m}$", 4, 1))
pgm.add_node(daft.Node("zsm1", r"$z_{*m}^1$", 4, 2))
pgm.add_node(daft.Node("zsm2", r"$z_{*m}^{L-1}$", 4, 3))


pgm.add_edge("xnm", "hnm1")
pgm.add_edge("hnm1", "hnm2")

pgm.add_edge("hnm1", "znm1")
pgm.add_edge("hnm2", "znm2")

pgm.add_edge("xnm", "Lnmx")
pgm.add_edge("znm1", "Lnm1")
pgm.add_edge("znm2", "Lnm2")

pgm.add_edge("xsm", "Lnmx")
pgm.add_edge("zsm1", "Lnm1")
pgm.add_edge("zsm2", "Lnm2")

pgm.add_edge("zsm2", "zsm1")
pgm.add_edge("zsm1", "xsm")

##

pgm.add_node(daft.Node("yn", r"$y_{n}$", 8, 1))
pgm.add_node(daft.Node("hnL", r"$h_{n}^L$", 8, 4))
pgm.add_node(daft.Node("znL", r"$z_{n}^L$", 7, 4))

pgm.add_node(daft.Node("Lny", r"$\calL_{n}^y$", 6, 1))
pgm.add_node(daft.Node("LnL", r"$\calL_{n}^L$", 6, 4))

pgm.add_node(daft.Node("zsL", r"$z_{*}^L$", 5, 4))
pgm.add_node(daft.Node("ys", r"$y_{*}$", 5, 1))



pgm.add_edge("yn", "hnL")
pgm.add_edge("hnL", "znL")

pgm.add_edge("yn", "Lny")
pgm.add_edge("znL", "LnL")

pgm.add_edge("ys", "Lny")
pgm.add_edge("zsL", "LnL")

pgm.add_edge("zsL", "ys")

pgm.add_edge("zsL", "zsm2")

###

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "infNetCrossModalY"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

import daft
import os

pgm = daft.PGM([10, 6], origin=[0, 0])

pgm.add_node(daft.Node("xnm", r"$x_{nm}$", 1, 1))
pgm.add_node(daft.Node("hnm1", r"$h_{nm}^1$", 1, 2))
pgm.add_node(daft.Node("hnmL", r"$h_{nm}^{L}$", 1, 3))

pgm.add_node(daft.Node("znm1", r"$z_{nm}^1$", 2, 2))
pgm.add_node(daft.Node("znmL", r"$z_{nm}^{L}$", 2, 3))

pgm.add_node(daft.Node("Lnmx", r"$\calL_{nm}^x$", 3, 1))
pgm.add_node(daft.Node("Lnm1", r"$\calL_{nm}^1$", 3, 2))
pgm.add_node(daft.Node("LnmL", r"$\calL_{nm}^L$", 3, 3))

pgm.add_node(daft.Node("xsm", r"$\tilde{x}_{nm}$", 4, 1))
pgm.add_node(daft.Node("zsm1", r"$\tilde{z}_{nm}^1$", 4, 2))
pgm.add_node(daft.Node("zsmL", r"$\tilde{z}_{nm}^{L}$", 4, 3))


pgm.add_edge("xnm", "hnm1")
pgm.add_edge("hnm1", "hnmL")

pgm.add_edge("hnm1", "znm1")
pgm.add_edge("hnmL", "znmL")

pgm.add_edge("xnm", "Lnmx")
pgm.add_edge("znm1", "Lnm1")
pgm.add_edge("znmL", "LnmL")

pgm.add_edge("xsm", "Lnmx")
pgm.add_edge("zsm1", "Lnm1")
pgm.add_edge("zsmL", "LnmL")

pgm.add_edge("zsmL", "zsm1")
pgm.add_edge("zsm1", "xsm")

###

pgm.add_node(daft.Node("xno", r"$x_{no}$", 9, 1))
pgm.add_node(daft.Node("hno1", r"$h_{no}^1$", 9, 2))
pgm.add_node(daft.Node("hnoL", r"$h_{no}^{L}$", 9, 3))

pgm.add_node(daft.Node("zno1", r"$z_{no}^1$", 8, 2))
pgm.add_node(daft.Node("znoL", r"$z_{no}^{L}$", 8, 3))

pgm.add_node(daft.Node("Lnox", r"$\calL_{no}^x$", 7, 1))
pgm.add_node(daft.Node("Lno1", r"$\calL_{no}^1$", 7, 2))
pgm.add_node(daft.Node("LnoL", r"$\calL_{no}^L$", 7, 3))

pgm.add_node(daft.Node("xso", r"$\tilde{x}_{no}$", 6, 1))
pgm.add_node(daft.Node("zso1", r"$\tilde{z}_{no}^1$", 6, 2))
pgm.add_node(daft.Node("zsoL", r"$\tilde{z}_{no}^{L}$", 6, 3))

pgm.add_edge("xno", "hno1")
pgm.add_edge("hno1", "hnoL")

pgm.add_edge("hno1", "zno1")
pgm.add_edge("hnoL", "znoL")

pgm.add_edge("xno", "Lnox")
pgm.add_edge("zno1", "Lno1")
pgm.add_edge("znoL", "LnoL")

pgm.add_edge("xso", "Lnox")
pgm.add_edge("zso1", "Lno1")
pgm.add_edge("zsoL", "LnoL")

pgm.add_edge("zsoL", "zso1")
pgm.add_edge("zso1", "xso")

###

pgm.add_node(daft.Node("zs", r"$\tilde{z}_{n}$", 5, 4))
pgm.add_edge("zs", "zsmL")
pgm.add_edge("zs", "zsoL")

pgm.add_node(daft.Node("LL", r"$\calL_{n,o,m}$", 5, 5))
pgm.add_edge("znmL", "LL")
pgm.add_edge("znoL", "LL")
#pgm.add_edge("znoL", "LL", label="My label")



###



pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "infNetCrossModal"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


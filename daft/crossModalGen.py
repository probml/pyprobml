from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

import daft
import os

pgm = daft.PGM([3, 5], origin=[0, 0])


pgm.add_node(daft.Node("xnm", r"$x_{nm}$", 1, 1))
pgm.add_node(daft.Node("znm1", r"$z_{nm}^1$", 1, 2))
pgm.add_node(daft.Node("znmL", r"$z_{nm}^{L}$", 1, 3))
pgm.add_node(daft.Node("znL", r"$z_{n}$", 1, 4))

pgm.add_edge("znL", "znmL")
pgm.add_edge("znmL", "znm1")
pgm.add_edge("znm1", "xnm")

pgm.add_plate(daft.Plate([0.5, 0.5, 1.2, 3], label=r"$m=0:M$", shift=-0.1))
pgm.add_plate(daft.Plate([0.2, 0.2, 2, 4.2], label=r"$n=1:N$", shift=-0.1))

#pgm.add_node(daft.Node("xnm", r"$x_{nm}$", 1, 1))
#pgm.add_node(daft.Node("znm1", r"$z_{nm}^1$", 1, 2))
#pgm.add_node(daft.Node("znm2", r"$z_{nm}^{L-1}$", 1, 3))
#pgm.add_node(daft.Node("znL", r"$z_{n}^{L}$", 2, 4))
#pgm.add_node(daft.Node("yn", r"$y_{n}$", 2, 1))
#
#pgm.add_edge("znL", "yn")
#pgm.add_edge("znL", "znm2")
#pgm.add_edge("znm2", "znm1")
#pgm.add_edge("znm1", "xnm")
#
#pgm.add_plate(daft.Plate([0.5, 0.5, 1, 3], label=r"$m=1:M$", shift=-0.1))
#pgm.add_plate(daft.Plate([0.2, 0.2, 2.5, 4.5], label=r"$n=1:N$", shift=-0.1))



pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "CrossModalGen"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


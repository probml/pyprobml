from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

import daft
import os

pgm = daft.PGM([5, 8], origin=[0, 0])


pgm.add_node(daft.Node("x", r"$x$", 1, 1))
pgm.add_node(daft.Node("h1", r"$h^1$", 1, 3))
pgm.add_node(daft.Node("h2", r"$h^{L-1}$", 1, 5))
pgm.add_node(daft.Node("h3", r"$h^L$", 1, 7))

pgm.add_node(daft.Node("lam1", r"$\lambda^1$", 2, 3))
pgm.add_node(daft.Node("lam2", r"$\lambda^{L-1}$", 2, 5))
pgm.add_node(daft.Node("lam3", r"$\lambda^L$", 2, 7))

pgm.add_node(daft.Node("gam1", r"$\gamma^1$", 3, 3))
pgm.add_node(daft.Node("gam2", r"$\gamma^{L-1}$", 3, 5))
pgm.add_node(daft.Node("gam3", r"$\gamma^L$", 3, 7))

pgm.add_node(daft.Node("pi1", r"$\pi^1$", 4, 3))
pgm.add_node(daft.Node("pi2", r"$\pi^{L-1}$", 4, 5))
pgm.add_node(daft.Node("pi3", r"$\pi^L$", 4, 7))

pgm.add_node(daft.Node("L0", r"$\calL^0$", 3, 1))
pgm.add_node(daft.Node("L1", r"$\calL^1$", 3, 2))
pgm.add_node(daft.Node("L2", r"$\calL^{L-1}$", 3, 4))
pgm.add_node(daft.Node("L3", r"$\calL^L$", 3, 6))


pgm.add_node(daft.Node("z1", r"$z^1$", 4, 2))
pgm.add_node(daft.Node("z2", r"$z^{L-1}$", 4, 4))
pgm.add_node(daft.Node("z3", r"$z^L$", 4, 6))

pgm.add_edge("x", "h1")
pgm.add_edge("h1", "h2")
pgm.add_edge("h2", "h3")

pgm.add_edge("h1", "lam1")
pgm.add_edge("h2", "lam2")
pgm.add_edge("h3", "lam3")

pgm.add_edge("lam1", "gam1")
pgm.add_edge("lam2", "gam2")
pgm.add_edge("lam3", "gam3")

pgm.add_edge("pi1", "gam1")
pgm.add_edge("pi2", "gam2")
#pgm.add_edge("pi3", "gam3")

pgm.add_edge("gam1", "z1")
pgm.add_edge("gam2", "z2")
pgm.add_edge("gam3", "z3")

pgm.add_edge("z3", "pi2")
pgm.add_edge("z2", "pi1")

pgm.add_edge("gam1", "L1")
pgm.add_edge("gam2", "L2")
pgm.add_edge("gam3", "L3")

pgm.add_edge("pi1", "L1")
pgm.add_edge("pi2", "L2")
pgm.add_edge("pi3", "L3")

pgm.add_edge("x", "L0")
pgm.add_edge("z1", "L0")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "LVAE"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))



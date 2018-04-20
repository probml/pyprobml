
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

import daft
import os

pgm = daft.PGM([6, 4], origin=[0, 0])

pgm.add_node(daft.Node("S0", r"$\post{\vS}_k$", 1, 1, fixed=True))
pgm.add_node(daft.Node("v0", r"$\post{\nu}_k$", 1, 1.5, fixed=True))
pgm.add_node(daft.Node("k0", r"$\post{\kappa}_k$", 1, 2, fixed=True))
pgm.add_node(daft.Node("m0", r"$\post{\vm}_k$", 1, 2.5, fixed=True))

pgm.add_node(daft.Node("Lam_k", r"$\vLambda_k$", 3, 1))
pgm.add_node(daft.Node("mu_k", r"$\vmu_k$", 3, 2))

pgm.add_node(daft.Node("alpha0", r"$\post{\valpha}$", 1, 3.5, fixed=True))
pgm.add_node(daft.Node("pi", r"$\vpi$", 5, 3.5))

pgm.add_node(daft.Node("zz_n", r"$\post{\vz}_n$", 5, 1, \
    offset=(-15,-10), fixed=True))
pgm.add_node(daft.Node("z_n", r"$\vz_n$", 5, 2))


pgm.add_plate(daft.Plate([0.5, 0.5, 3, 2.5], label=r"$k=1:K$", shift=-0.1))
pgm.add_plate(daft.Plate([4.5, 0.5, 1, 2], label=r"$n=1:N$", shift=-0.1))



pgm.add_edge("S0", "Lam_k")
pgm.add_edge("v0", "Lam_k")
pgm.add_edge("k0", "mu_k")
pgm.add_edge("m0", "mu_k")
pgm.add_edge("alpha0", "pi")

pgm.add_edge("Lam_k", "mu_k")

pgm.add_edge("zz_n", "z_n")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "GMM-DGM-var"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


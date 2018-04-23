"""
Yike's model
============

This is Yike Tang's model for weak lensing.

"""

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

pgm = daft.PGM([5.20, 2.95], origin=[-1.70, 1.65])
pgm.add_node(daft.Node("obs", r"$\epsilon^{\mathrm{obs}}_n$", 2, 3, observed=True))
pgm.add_node(daft.Node("true", r"$\epsilon^{\mathrm{true}}_n$", 1, 3))
pgm.add_edge("true", "obs")
pgm.add_node(daft.Node("alpha", r"$\alpha,\beta$", -0.25, 3))
pgm.add_edge("alpha", "true")
pgm.add_node(daft.Node("shape prior", r"$p(\alpha, \beta)$", -1.25, 3, fixed=True))
pgm.add_edge("shape prior", "alpha")
pgm.add_node(daft.Node("gamma", r"$\gamma_m$", 2, 4))
pgm.add_edge("gamma", "obs")
pgm.add_node(daft.Node("gamma prior", r"$p(\gamma)$", -0.25, 4, fixed=True))
pgm.add_edge("gamma prior", "gamma")
pgm.add_node(daft.Node("sigma", r"$\sigma_{\epsilon}$", 3.25, 3, fixed=True))
pgm.add_edge("sigma", "obs")
pgm.add_plate(daft.Plate([0.5, 2.25, 2, 1.25],
        label=r"galaxies $n$"))
pgm.add_plate(daft.Plate([0.25, 1.75, 2.5, 2.75],
        label=r"patches $m$"))
pgm.render()
pgm.figure.savefig("yike.pdf")
pgm.figure.savefig("yike.png", dpi=150)

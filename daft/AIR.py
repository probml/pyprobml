#Attend, Infer, Repeat

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

import daft
import os



pgm = daft.PGM([5, 6], origin=[1.5, 0.5], node_unit=1.5)

pgm.add_node(daft.Node("x", r"$x$",3, 1))
pgm.add_node(daft.Node("xt1", r"$x^d_{1:l-1}$", 2, 2))
pgm.add_node(daft.Node("xt", r"$x^d_{1:l}$", 5, 2))
pgm.add_node(daft.Node("dt", r"$\Delta_{l}^e$", 3, 3))
pgm.add_node(daft.Node("atr", r"$z_l^{\mathrm{where},e}$", 3, 4))
pgm.add_node(daft.Node("rt", r"$z_l^{\mathrm{what},e}$", 4, 4))
pgm.add_node(daft.Node("wt", r"$z_{l}^{\mathrm{what},d}$", 6, 4))
pgm.add_node(daft.Node("it", r"$\Delta_{l}^d$", 5, 3))
pgm.add_node(daft.Node("atw", r"$z_l^{\mathrm{where},d}$", 5, 4))
pgm.add_node(daft.Node("ht1", r"$h_{l-1}$", 2, 5))
pgm.add_node(daft.Node("ht", r"$h_{l}$", 5, 5))
#pgm.add_node(daft.Node("zt", r"$z_{l}$", 5, 6))

pgm.add_edge("x", "dt")
pgm.add_edge("x", "rt")
pgm.add_edge("xt1", "xt")
pgm.add_edge("xt1", "dt")
pgm.add_edge("ht1", "ht")
pgm.add_edge("dt", "atr")
pgm.add_edge("atr", "rt")
pgm.add_edge("rt", "ht")
#pgm.add_edge("ht1", "zt")
#pgm.add_edge("zt", "ht")
pgm.add_edge("ht", "wt")
pgm.add_edge("ht", "atw")
pgm.add_edge("wt", "it")
pgm.add_edge("atw", "it")
pgm.add_edge("it", "xt")
pgm.add_edge("atr", "ht")

pgm.render()
fname = "AIR-inf"
folder = "/Users/kpmurphy/github/pyprobml/figures"
#pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


############

pgm = daft.PGM([4, 5], origin=[4.5, 1.5], node_unit=1.5)


pgm.add_node(daft.Node("xt", r"$x_{1:l}$", 5, 2))
pgm.add_node(daft.Node("wt", r"$z_l^{\mathrm{what}}$", 6, 4))
pgm.add_node(daft.Node("it", r"$\Delta_{l}$", 5, 3))
pgm.add_node(daft.Node("atw", r"$z_l^{\mathrm{where}}$", 5, 4))
pgm.add_node(daft.Node("ht", r"$h_{l}$", 5, 5))
#pgm.add_node(daft.Node("zt", r"$z_{l}$", 5, 6))

pgm.add_node(daft.Node("xtt", r"$x_{1:l+1}$", 7, 2))
pgm.add_node(daft.Node("wtt", r"$z_{l+1}^{\mathrm{what}}$", 8, 4))
pgm.add_node(daft.Node("itt", r"$\Delta_{l+1}$", 7, 3))
pgm.add_node(daft.Node("attw", r"$z_{l+1}^{\mathrm{where}}$", 7, 4))
pgm.add_node(daft.Node("htt", r"$h_{l+1}$", 7, 5))
#pgm.add_node(daft.Node("ztt", r"$z_{l+1}$", 7, 6))



#pgm.add_edge("zt", "ht")
pgm.add_edge("ht", "wt")
pgm.add_edge("ht", "atw")
pgm.add_edge("wt", "it")
pgm.add_edge("atw", "it")
pgm.add_edge("it", "xt")

pgm.add_edge("xt", "xtt")
pgm.add_edge("ht", "htt")

#pgm.add_edge("ztt", "htt")
pgm.add_edge("htt", "wtt")
pgm.add_edge("htt", "attw")
pgm.add_edge("wtt", "itt")
pgm.add_edge("attw", "itt")
pgm.add_edge("itt", "xtt")

pgm.add_edge("wt", "htt")
pgm.add_edge("atw", "htt")

pgm.render()
#folder = "figures"
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "AIR-gen"
#pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


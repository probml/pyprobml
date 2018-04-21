"""
T-shirt style
=============

Don't like dainty thin lines?  Need to make graphical-model-themed
conference schwag?  Then `line_width` is the parameter for you.  Also
check out the `preamble` option in the `matplotlib.rc` command.

"""

from matplotlib import rc
rc("font", family="serif", size=14)
rc("text", usetex=True)
rc('text.latex',
   preamble="\usepackage{amssymb}\usepackage{amsmath}\usepackage{mathrsfs}")

import daft

# Instantiate the PGM.
pgm = daft.PGM([2.3, 2.05], origin=[0.3, 0.3], line_width=2.5)

# Hierarchical parameters.
pgm.add_node(daft.Node("alpha", r"$\boldsymbol{\alpha}$", 0.5, 2, fixed=True))
pgm.add_node(daft.Node("beta", r"$\boldsymbol{\beta}$", 1.5, 2))

# Latent variable.
pgm.add_node(daft.Node("w", r"$\boldsymbol{w_n}$", 1, 1))

# Data.
pgm.add_node(daft.Node("x", r"$\boldsymbol{x_n}$", 2, 1, observed=True))

# Add in the edges.
pgm.add_edge("alpha", "beta")
pgm.add_edge("beta", "w")
pgm.add_edge("w", "x")
pgm.add_edge("beta", "x")

# And a plate.
pgm.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$\boldsymbol{n = 1, \cdots, N}$",
    shift=-0.1))

# Render and save.
pgm.render()
pgm.figure.savefig("thicklines.pdf")
pgm.figure.savefig("thicklines.png", dpi=150)

# Object oriented Generative image model
# oo-gen-obs
# oo-gen-dynamics
# oo-inf-loc
# oo-inf-state

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

import daft
import os


pgm = daft.PGM([4, 5], origin=[0, 0], node_unit=1.5)
pgm.add_node(daft.Node("x", r"$x$",3, 1))

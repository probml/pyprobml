import os

import imp
import daft
#daft = imp.load_source('daft', 'daft-080308/daft.py')

pgm = daft.PGM([9, 6], origin=[0, 0], observed_style="inner")

pgm.add_node(daft.Node("Z1", r"$Z_1$", 2, 5))
pgm.add_node(daft.Node("Z2", r"$Z_2$", 3, 5))
pgm.add_node(daft.Node("Z3", r"$Z_3$", 4, 5))
pgm.add_node(daft.Node("F", r"$F$", 3, 1))
pgm.add_node(daft.Node("Y1", r"$Y_1$", 2, 2))
pgm.add_node(daft.Node("Y2", r"$Y_2$", 4, 2))
pgm.add_node(daft.Node("C", r"$C$", 1, 3))
pgm.add_node(daft.Node("X1", r"$X_1$", 2, 4))
pgm.add_node(daft.Node("I1", r"$I_1$", 3, 4))
pgm.add_node(daft.Node("X2", r"$X_2$", 4, 4))
pgm.add_node(daft.Node("I2", r"$I_2$", 5, 4))
pgm.add_node(daft.Node("N", r"$N=2$", 1, 4))

pgm.add_edge("Z1", "X1")
pgm.add_edge("Z2", "X1")
pgm.add_edge("Z3", "X1")
pgm.add_edge("Z1", "X2")
pgm.add_edge("Z2", "X2")
pgm.add_edge("Z3", "X2")

pgm.add_edge("I1", "X1")
pgm.add_edge("I2", "X2")
pgm.add_edge("X1", "Y1")
pgm.add_edge("X2", "Y2")
pgm.add_edge("C", "Y1")
pgm.add_edge("C", "Y2")
pgm.add_edge("Y1", "F")
pgm.add_edge("Y2", "F")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "multi-obj-tracker-gen"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
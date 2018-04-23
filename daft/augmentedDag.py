from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())
#rc("text.latex", preamble=open("examples/daft/macros.tex").read())

import daft
#import imp
#daft = imp.load_source('daft', 'dfm-daft-6038869/daft.py')
 
import os

pgm = daft.PGM([2, 2], origin=[0.5, 0.5])

pgm.add_node(daft.Node("A", r"$A$", 1, 1))
pgm.add_node(daft.Node("B", r"$B$", 2, 1))

pgm.add_node(daft.Node("FA", r"$F_A$", 1, 2))
pgm.add_node(daft.Node("FB", r"$F_B$", 2, 2))

pgm.add_edge("FA", "A")
pgm.add_edge("FB", "B")
pgm.add_edge("A", "B")

pgm.render()
folder = "../../figures"
fname = "augmentedDagAtoB"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))




pgm = daft.PGM([2, 2], origin=[0.5, 0.5])

pgm.add_node(daft.Node("A", r"$A$", 1, 1))
pgm.add_node(daft.Node("B", r"$B$", 2, 1))

pgm.add_node(daft.Node("FA", r"$F_A$", 1, 2))
pgm.add_node(daft.Node("FB", r"$F_B$", 2, 2))

pgm.add_edge("FA", "A")
pgm.add_edge("FB", "B")
pgm.add_edge("B", "A")

pgm.render()
folder = "../../figures"
fname = "augmentedDagBtoA"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))

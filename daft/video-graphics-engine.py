# figures for structured video models

#from matplotlib import rc
#rc("font", family="serif", size=12)
#rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())

#import daft
import os

import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')

pgm = daft.PGM([8, 9], origin=[0, 0], observed_style="inner")

x = 2
pgm.add_node(daft.Node("LIt1", r"$L_{t-1}^I$", x, 1, observed=True))
pgm.add_node(daft.Node("It1", r"$I_{t-1}$", x-1, 2))
pgm.add_node(daft.Node("Ipredt1", r"$\hat{I}_{t-1}$", x+1, 2, observed=True))
pgm.add_node(daft.Node("et1", r"se", x-1, 3))
pgm.add_node(daft.Node("gt1", r"ge", x+1, 3))
#pgm.add_node(daft.Node("Ot1", r"$O_{t-1}$", x, 4, shape="ellipse"))
pgm.add_node(daft.Node("Ot1", r"$O_{t-1}$", x, 4.0, shape="rectangle"))
pgm.add_node(daft.Node("LSt1", r"$L_{t-1}^S$", x, 5, observed=True))
pgm.add_node(daft.Node("St1", r"$S_{t-1}$", x, 6, observed=True))

pgm.add_edge("It1", "LIt1")
pgm.add_edge("Ipredt1", "LIt1")
pgm.add_edge("It1", "et1")
pgm.add_edge("gt1", "Ipredt1")
pgm.add_edge("et1", "Ot1")
pgm.add_edge("Ot1", "LSt1")
pgm.add_edge("Ot1", "gt1")
pgm.add_edge("St1", "LSt1")

x = 6
pgm.add_node(daft.Node("LIt", r"$L_{t}^I$", x, 1, observed=True))
pgm.add_node(daft.Node("It", r"$I_{t}$", x-1, 2))
pgm.add_node(daft.Node("Ipredt", r"$\hat{I}_{t}$", x+1, 2, observed=True))
pgm.add_node(daft.Node("et", r"se", x-1, 3))
pgm.add_node(daft.Node("gt", r"ge", x+1, 3))
pgm.add_node(daft.Node("Ot", r"$O_{t}$", x, 4.0, shape="rectangle"))
pgm.add_node(daft.Node("LSt", r"$L_{t}^S$", x, 5, observed=True))
pgm.add_node(daft.Node("St", r"$S_{t}$", x, 6, observed=True))

pgm.add_edge("It", "LIt")
pgm.add_edge("Ipredt", "LIt")
pgm.add_edge("It", "et")
pgm.add_edge("gt", "Ipredt")
pgm.add_edge("et", "Ot")
pgm.add_edge("Ot", "LSt")
pgm.add_edge("Ot", "gt")
pgm.add_edge("St", "LSt")

pgm.add_node(daft.Node("dyn", r"dyn", 4, 4))
pgm.add_edge("Ot1", "dyn")
pgm.add_edge("dyn", "Ot")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "video-graphics-engine"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))

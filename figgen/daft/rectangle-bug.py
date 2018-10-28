import os

import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')

pgm = daft.PGM([8, 9], origin=[0, 0], observed_style="inner")

x = 2
pgm.add_node(daft.Node("se", r"se", x-1, 3))
pgm.add_node(daft.Node("ge", r"ge", x+1, 3))
#pgm.add_node(daft.Node("Ot1", r"$O_{t-1}$", x, 4, shape="ellipse"))
pgm.add_node(daft.Node("O", r"$O_{t-1}$", x, 4.0, shape="rectangle"))

pgm.add_edge("se", "O")
pgm.add_edge("O", "ge")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "rectangle-bug"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))

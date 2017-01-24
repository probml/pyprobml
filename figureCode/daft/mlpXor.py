from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

#import daft
	
import imp
daft = imp.load_source('daft', '/Users/kpmurphy/github/daft/daft.py')
 
import os

pgm = daft.PGM([4, 4], origin=[0, 0])

pgm.add_node(daft.Node("one", r"$1$", 1, 1))
pgm.add_node(daft.Node("x1", r"$x_1$", 2, 1))
pgm.add_node(daft.Node("x2", r"$x_2$", 3, 1))

pgm.add_node(daft.Node("z1", r"$z_1$", 2, 2))
pgm.add_node(daft.Node("z2", r"$z_2$", 3, 2))

pgm.add_node(daft.Node("y", r"$y$", 2.5, 3))

pgm.add_edge("one", "z1", label="-1.5", xoffset=-0.3)
pgm.add_edge("x1", "z1", label="+1", xoffset=-0.3)
pgm.add_edge("x1", "z2", label="+1", xoffset=-0.4)
pgm.add_edge("x2", "z1", label="+1", xoffset=0.4)
pgm.add_edge("x2", "z2", label="+1", xoffset=0.3)
pgm.add_edge("z1", "y", label="-1", xoffset=-0.3)
pgm.add_edge("z2", "y", label="-1", xoffset=0.3)

#ax = pgm.render()  # returns the pyplot axes object it drew onto
#ax.text(1, 2, "My label")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "mlpXor"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))



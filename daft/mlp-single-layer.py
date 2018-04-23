from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())
#rc("text.latex", preamble=open("examples/daft/macros.tex").read())

#import daft
import imp
#daft = imp.load_source('daft', '/Users/kpmurphy/github/daft/daft.py')
#daft = imp.load_source('daft', 'dfm-daft-6038869/daft.py')
daft = imp.load_source('daft', 'daft-080308/daft.py')
 
import os

pgm = daft.PGM([5, 4], origin=[0, 0])

pgm.add_node(daft.Node("one", r"$1$", 1, 1))
pgm.add_node(daft.Node("one2", r"$1$", 4, 1))

pgm.add_node(daft.Node("x1", r"$x_1$", 2, 1))
pgm.add_node(daft.Node("x2", r"$x_2$", 3, 1))

pgm.add_node(daft.Node("z1", r"$z_1$", 2, 2))
pgm.add_node(daft.Node("z2", r"$z_2$", 3, 2))

pgm.add_node(daft.Node("y", r"$y$", 2.5, 3))

if 0:
    pgm.add_edge("one", "z1", label="$W_{1;01}$")
    pgm.add_edge("one2", "z2", label="$W^1_{02}$")
    pgm.add_edge("x1", "z1", label="$W^1_{11}$")
    pgm.add_edge("x1", "z2", label="$W^1_{12}$")
    pgm.add_edge("x2", "z1", label="$W^1_{21}$")
    pgm.add_edge("x2", "z2", label="$W^1_{22}$")
    pgm.add_edge("z1", "y", label="$W^2_{1}$")
    pgm.add_edge("z2", "y", label="$W^2_{2}$")

if 1:
    pgm.add_edge("one", "z1", label="$W^1_{01}$", xoffset=-0.5)
    pgm.add_edge("one2", "z2", label="$W^1_{02}$", xoffset=0.5)
    pgm.add_edge("x1", "z1", label="$W^1_{11}$", xoffset=-0.4, yoffset=-0.3)
    pgm.add_edge("x1", "z2", label="$W^1_{12}$", xoffset=-0.4)
    pgm.add_edge("x2", "z1", label="$W^1_{21}$", xoffset=0.4)
    pgm.add_edge("x2", "z2", label="$W^1_{22}$", xoffset=0.4, yoffset=-0.3)
    pgm.add_edge("z1", "y", label="$W^2_{1}$", xoffset=-0.5)
    pgm.add_edge("z2", "y", label="$W^2_{2}$", xoffset=0.5)

if 0:
    pgm.add_edge("one", "z1", label="b", xoffset=-0.3)
    pgm.add_edge("x1", "z1", label="w_{1,11}", xoffset=-0.3)
    pgm.add_edge("x1", "z2", label="w_{1,12}", xoffset=-0.4)
    pgm.add_edge("x2", "z1", label="w_{1,21}", xoffset=0.4)
    pgm.add_edge("x2", "z2", label="w_{1,22}", xoffset=0.3)
    pgm.add_edge("z1", "y", label="w_{2,11}", xoffset=-0.3)
    pgm.add_edge("z2", "y", label="w_{2,21}", xoffset=0.3)

#ax = pgm.render()  # returns the pyplot axes object it drew onto
#ax.text(1, 2, "My label")

pgm.render()
folder = "../../figures"
fname = "mlp-single-layer"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))



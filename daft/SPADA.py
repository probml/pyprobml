import imp
#import daft
import os
daft = imp.load_source('daft', 'daft-080308/daft.py')

pgm = daft.PGM([11, 11], origin=[0, 0], observed_style="shaded", directed=False)

pgm.add_node(daft.Node("xoldN", r"$z_{t-1}^N$", 1.0, 2.0, shape="ellipse"))
pgm.add_node(daft.Node("fN", r"$f^N$", 2.0, 2.0, shape="rectangle"))
pgm.add_node(daft.Node("xN", r"$z_{t}^N$", 3.0, 2.0, shape="ellipse"))


pgm.add_node(daft.Node("gN", r"$g^N$", 4.0, 4.0, shape="rectangle"))
pgm.add_node(daft.Node("aN", r"$a_{t}^N$", 5.0, 4.0, shape="ellipse"))

pgm.add_node(daft.Node("psiN1", r"$\Psi^{N,1}$", 6.0, 5.0, shape="rectangle"))
pgm.add_node(daft.Node("psiNM", r"$\Psi^{N,M}$", 6.0, 3.0, shape="rectangle"))

pgm.add_node(daft.Node("bM", r"$b_{t}^M$", 7.0, 4.0, shape="ellipse"))
pgm.add_node(daft.Node("hM", r"$h^{M}$", 8.0, 4.0, shape="rectangle"))
pgm.add_node(daft.Node("obsM", r"$x_{t}^M$", 9.0, 4.0, shape="ellipse", observed=True))

pgm.add_node(daft.Node("xold1", r"$z_{t-1}^1$", 1.0, 10.0, shape="ellipse"))
pgm.add_node(daft.Node("f1", r"$f^1$", 2.0, 10.0, shape="rectangle"))
pgm.add_node(daft.Node("x1", r"$z_{t}^1$", 3.0, 10.0, shape="ellipse"))

pgm.add_node(daft.Node("g1", r"$g^1$", 4.0, 8.0, shape="rectangle"))
pgm.add_node(daft.Node("a1", r"$a_{t}^1$", 5.0, 8.0, shape="ellipse"))

pgm.add_node(daft.Node("psi11", r"$\Psi^{1,1}$", 6.0, 9.0, shape="rectangle"))
pgm.add_node(daft.Node("psi1M", r"$\Psi^{1,M}$", 6.0, 7.0, shape="rectangle"))

pgm.add_node(daft.Node("b1", r"$b_{t}^1$", 7.0, 8.0, shape="ellipse"))
pgm.add_node(daft.Node("h1", r"$h^{1}$", 8.0, 8.0, shape="rectangle"))
pgm.add_node(daft.Node("obs1", r"$x_{t}^1$", 9.0, 8.0, shape="ellipse", observed=True))

pgm.add_node(daft.Node("obsAll", r"$x_{t}^{1:M}$", 9.5, 1.0, shape="ellipse", observed=True))
pgm.add_node(daft.Node("obsAllOld", r"$x_{t}^{1:M}$", 4.5, 1.0, shape="ellipse", observed=True))

pgm.add_edge("xoldN", "fN", directed=False)
pgm.add_edge("fN", "xN")
pgm.add_edge("xN", "gN")
pgm.add_edge("gN", "aN")
pgm.add_edge("aN", "psiN1")
pgm.add_edge("aN", "psiNM")
pgm.add_edge("psiN1", "b1")
pgm.add_edge("psiNM", "bM")
pgm.add_edge("bM", "hM")
pgm.add_edge("hM", "obsM")

pgm.add_edge("xold1", "f1")
pgm.add_edge("f1", "x1")
pgm.add_edge("x1", "g1")
pgm.add_edge("g1", "a1")
pgm.add_edge("a1", "psi11")
pgm.add_edge("a1", "psi1M")
pgm.add_edge("psi11", "b1")
pgm.add_edge("psi1M", "bM")
pgm.add_edge("b1", "h1")
pgm.add_edge("h1", "obs1")

pgm.add_edge("obsM", "obsAll")
pgm.add_edge("obs1", "obsAll")
pgm.add_edge("obsAll", "obsAllOld")
pgm.add_edge("gN", "obsAllOld")
pgm.add_edge("g1", "obsAllOld")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "SPADA"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))

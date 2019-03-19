from matplotlib import rc
import matplotlib.pyplot as plt
rc("font", family="serif", size=12)
rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())
#rc('text.latex',   preamble="\usepackage{amssymb}\usepackage{amsmath}\usepackage{mathrsfs}")
   
import os

import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')
#import daft



pgm = daft.PGM([7, 4], origin=[0, 0])

pgm.add_node(daft.Node("v1", r"$v_{t-1}$", 1, 1))
pgm.add_node(daft.Node("v2", r"$v_{t}$", 3, 1))
pgm.add_node(daft.Node("v3", r"$0$", 5, 1))
pgm.add_node(daft.Node("s1", r"$s_{t-1}$", 1, 3))
pgm.add_node(daft.Node("s2", r"$s_{t}$", 3, 3))
pgm.add_node(daft.Node("s3", r"$s_{t+1}$", 5, 3))
pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 2, 2))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 4, 2))
pgm.add_node(daft.Node("h3", r"$h_{t+1}$", 6, 2))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h2", "h3", linestyle="-")
#pgm.add_edge("v1", "s1", linestyle="-")
pgm.add_edge("v2", "s2", linestyle="-")
pgm.add_edge("v3", "s3", linestyle="-")
pgm.add_edge("v1", "h1", linestyle=":")
pgm.add_edge("v2", "h2", linestyle=":")
pgm.add_edge("v3", "h3", linestyle=":")
pgm.add_edge("s1", "h1", linestyle="-")
pgm.add_edge("s2", "h2", linestyle="-")
pgm.add_edge("s3", "h3", linestyle="-")
pgm.add_edge("h1", "s2", linestyle="-")
pgm.add_edge("h2", "s3", linestyle="-")
pgm.add_edge("s1", "s2", linestyle="-.")
pgm.add_edge("s2", "s3", linestyle="-.")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "visualVRNN"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


###


pgm = daft.PGM([7, 5], origin=[0, 0])

pgm.add_node(daft.Node("v1", r"$v_{t-1}$", 1, 1))
pgm.add_node(daft.Node("v2", r"$v_{t}$", 3, 1))
pgm.add_node(daft.Node("v3", r"$0$", 5, 1))
pgm.add_node(daft.Node("s1", r"$s_{t-1}$", 1, 4))
pgm.add_node(daft.Node("s2", r"$s_{t}$", 3, 4))
pgm.add_node(daft.Node("s3", r"$s_{t+1}$", 5, 4))
pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 2, 2))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 4, 2))
pgm.add_node(daft.Node("h3", r"$h_{t+1}$", 6, 2))
pgm.add_node(daft.Node("z1", r"$z_{t}$", 2, 3))
pgm.add_node(daft.Node("z2", r"$z_{t+1}$", 4, 3))
pgm.add_node(daft.Node("z3", r"$z_{t+2}$", 6, 3))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h2", "h3", linestyle="-")
#pgm.add_edge("v1", "s1", linestyle="-")
pgm.add_edge("v2", "s2", linestyle="-")
pgm.add_edge("v3", "s3", linestyle="-")
pgm.add_edge("v1", "h1", linestyle=":")
pgm.add_edge("v2", "h2", linestyle=":")
pgm.add_edge("v3", "h3", linestyle=":")
pgm.add_edge("s1", "h1", linestyle="-")
pgm.add_edge("s2", "h2", linestyle="-")
pgm.add_edge("s3", "h3", linestyle="-")
pgm.add_edge("h1", "s2", linestyle="-")
pgm.add_edge("h2", "s3", linestyle="-")
pgm.add_edge("s1", "s2", linestyle="-.")
pgm.add_edge("s2", "s3", linestyle="-.")

pgm.add_edge("h1", "z1", linestyle="-",)
pgm.add_edge("h2", "z2", linestyle="-")
pgm.add_edge("h3", "z3", linestyle="-")
pgm.add_edge("z1", "s2", linestyle="-")
pgm.add_edge("z2", "s3", linestyle="-")
pgm.add_edge("z1", "h2", linestyle="-")
pgm.add_edge("z2", "h3", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "visualVRNN2"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


###

pgm = daft.PGM([8, 8], origin=[0, -2])

pgm.add_node(daft.Node("h1A", r"$h_{t-1}^1$", 1, 2))
pgm.add_node(daft.Node("h1B", r"$h_{t-1}^2$", 1, 4))
pgm.add_node(daft.Node("hh1A", r"$\tilde{h}_{t-1}^1$", 2, 2))
pgm.add_node(daft.Node("hh1B", r"$\tilde{h}_{t-1}^2$", 2, 4))
pgm.add_node(daft.Node("hhh1A", r"$\hat{h}_{t}^1$", 5, 2))
pgm.add_node(daft.Node("hhh1B", r"$\hat{h}_{t}^2$", 5, 4))
pgm.add_node(daft.Node("h2A", r"$h_{t}^1$", 6, 2))
pgm.add_node(daft.Node("h2B", r"$h_{t}^2$", 6, 4))
pgm.add_node(daft.Node("xA", r"$x_{t}^1$", 3, 1))
pgm.add_node(daft.Node("xB", r"$x_{t}^2$", 3, 3))
pgm.add_node(daft.Node("zA", r"$z_{t}^1$", 4, 1))
pgm.add_node(daft.Node("zB", r"$z_{t}^2$", 4, 3))

pgm.add_edge("h1A", "hh1A", linestyle="-")
pgm.add_edge("h1B", "hh1B", linestyle="-")
pgm.add_edge("h1A", "hh1B", linestyle="-")
pgm.add_edge("h1B", "hh1A", linestyle="-")
pgm.add_edge("hhh1A", "h2B", linestyle="-")
pgm.add_edge("hhh1B", "h2A", linestyle="-")
pgm.add_edge("hh1A", "hhh1A", linestyle="-")
pgm.add_edge("hh1B", "hhh1B", linestyle="-")
pgm.add_edge("hhh1A", "h2A", linestyle="-")
pgm.add_edge("hhh1B", "h2B", linestyle="-")
pgm.add_edge("xA", "hhh1A", linestyle="-")
pgm.add_edge("xB", "hhh1B", linestyle="-")

pgm.add_edge("xA", "zA", linestyle="-")
pgm.add_edge("xB", "zB", linestyle="-")
pgm.add_edge("zA", "hhh1A", linestyle="-")
pgm.add_edge("zB", "hhh1B", linestyle="-")
pgm.add_edge("hh1A", "zA", linestyle="-")
pgm.add_edge("hh1B", "zB", linestyle="-")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "GVRNN"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


#input()


pgm = daft.PGM([6, 5], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 1, 2))
pgm.add_node(daft.Node("pz", r"$p_{t|t-1}^z$", 2, 3))
pgm.add_node(daft.Node("qz", r"$q_{t|t}^z$", 3, 3))
pgm.add_node(daft.Node("px", r"$p_{t|t-1}^x$", 4, 3))
pgm.add_node(daft.Node("x", r"$x_{t}$", 3, 1))
pgm.add_node(daft.Node("z", r"$z_{t|t}$", 4, 1))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 5, 2))
pgm.add_node(daft.Node("KL", r"$KL_{t}$", 3, 4))
pgm.add_node(daft.Node("LL", r"$LL_{t}$", 4, 4))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h1", "pz", linestyle="-")
pgm.add_edge("h1", "qz", linestyle="-")
pgm.add_edge("h1", "px", linestyle="-")
pgm.add_edge("x", "qz", linestyle="-")
pgm.add_edge("x", "h2", linestyle="-")
pgm.add_edge("z", "h2", linestyle="-")
pgm.add_edge("qz", "z", linestyle="-")
pgm.add_edge("z", "px", linestyle="-")
pgm.add_edge("pz", "KL", linestyle="-")
pgm.add_edge("qz", "KL", linestyle="-")
pgm.add_edge("px", "LL", linestyle="-")
pgm.add_edge("x", "LL", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "VRNNtrain"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


###


pgm = daft.PGM([6, 5], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 1, 2))
pgm.add_node(daft.Node("pz", r"$p_{t|t-1}^z$", 2, 3))
pgm.add_node(daft.Node("qz", r"$q_{t|t}^z$", 4, 3))
pgm.add_node(daft.Node("px", r"$p_{t|t-1}^x$", 3, 3))
pgm.add_node(daft.Node("x", r"$x_{t}$", 3, 1))
pgm.add_node(daft.Node("z", r"$z^*_{t}$", 4, 1))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 5, 2))
pgm.add_node(daft.Node("KL", r"$KL_{t}$", 2.5, 4))
pgm.add_node(daft.Node("LL", r"$LL_{t}$", 3.5, 4))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h1", "qz", linestyle="-")
pgm.add_edge("h1", "px", linestyle="-")
pgm.add_edge("h1", "pz", linestyle="-")
pgm.add_edge("z", "px", linestyle="-")
pgm.add_edge("x", "qz", linestyle="-")
pgm.add_edge("x", "h2", linestyle="-")
pgm.add_edge("z", "h2", linestyle="-")
pgm.add_edge("qz", "z", linestyle="-")
pgm.add_edge("pz", "KL", linestyle="-")
pgm.add_edge("qz", "KL", linestyle="-")
pgm.add_edge("px", "LL", linestyle="-")
pgm.add_edge("x", "LL", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "VRNNtrain2"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


###

###


pgm = daft.PGM([6, 4], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 1, 2))
pgm.add_node(daft.Node("pz", r"$p_{t|t-1}^z$", 2, 3))
pgm.add_node(daft.Node("qz", r"$q_{t|t}^z$", 4, 3))
pgm.add_node(daft.Node("px", r"$p_{t|t-1}^x$", 3, 3))
pgm.add_node(daft.Node("x", r"$x_{t}$", 3, 1))
pgm.add_node(daft.Node("z", r"$z^*_{t}$", 4, 1))
pgm.add_node(daft.Node("zprior", r"$\tilde{z}_{t}$", 2, 1))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 5, 2))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h1", "qz", linestyle="-")
pgm.add_edge("h1", "px", linestyle="-")
pgm.add_edge("h1", "pz", linestyle="-")
pgm.add_edge("pz", "zprior", linestyle="-")
pgm.add_edge("zprior", "px", linestyle="-")
pgm.add_edge("px", "x", linestyle="-")
pgm.add_edge("x", "qz", linestyle="-")
pgm.add_edge("x", "h2", linestyle="-")
pgm.add_edge("z", "h2", linestyle="-")
pgm.add_edge("qz", "z", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "VRNNtest"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


###


pgm = daft.PGM([5, 3], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 1, 2))
pgm.add_node(daft.Node("x", r"$x_{t}$", 2, 1))
pgm.add_node(daft.Node("z", r"$z_t$", 3, 1))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 4, 2))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("x", "h2", linestyle="-")
pgm.add_edge("z", "h2", linestyle="-")
pgm.add_edge("h1", "z", linestyle="-")
pgm.add_edge("x", "z", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "VRNNsimple"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))



pgm = daft.PGM([5, 4], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 1, 2))
pgm.add_node(daft.Node("px", r"$p_{t|t-1}^x$", 2, 3))
pgm.add_node(daft.Node("x", r"$x_{t}$", 2, 1))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 3, 2))
pgm.add_node(daft.Node("LL", r"$LL_{t}$", 2.5, 4))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h1", "px", linestyle="-")
pgm.add_edge("x", "h2", linestyle="-")
pgm.add_edge("px", "LL", linestyle="-")
pgm.add_edge("x", "LL", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "RNNtrain"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))





pgm = daft.PGM([5, 4], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 1, 2))
pgm.add_node(daft.Node("px", r"$p_{t|t-1}^x$", 2, 3))
pgm.add_node(daft.Node("x", r"$x_{t}$", 2, 1))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 3, 2))
pgm.add_node(daft.Node("LL", r"$LL_{t}$", 3, 3))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h1", "px", linestyle="-")
pgm.add_edge("x", "h2", linestyle="-")
pgm.add_edge("px", "LL", linestyle="-")
pgm.add_edge("x", "LL", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "RNNtrain"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))



pgm = daft.PGM([5, 4], origin=[0, 0])

pgm.add_node(daft.Node("h1", r"$h_{t-1}$", 1, 2))
pgm.add_node(daft.Node("px", r"$p_{t|t-1}^x$", 2, 3))
pgm.add_node(daft.Node("x", r"$x_{t}$", 2, 1))
pgm.add_node(daft.Node("h2", r"$h_{t}$", 3, 2))

pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h1", "px", linestyle="-")
pgm.add_edge("x", "h2", linestyle="-")
pgm.add_edge("px", "x", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "RNNtest"
pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))

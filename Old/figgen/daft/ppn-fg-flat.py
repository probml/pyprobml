
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())

import os

import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')
import daft



# Filter and predict

pgm = daft.PGM([5, 6], origin=[0, -1])

pgm.add_node(daft.Node("h0", r"$h_{t-1}$", 1, 3))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 2, 4))
pgm.add_node(daft.Node("FU1", r"$F_u$", 2.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("FP1", r"$F_p$", 2.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("FD1", r"$F_d$", 3.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("FDt1", r"$F_d$", 4.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("h1", r"$h_{t}$", 3, 3))
pgm.add_node(daft.Node("ht1", r"$\tilde{h}_{t}$", 4, 2))
pgm.add_node(daft.Node("p1", r"$p_{t}$", 3, 0))
pgm.add_node(daft.Node("pt1", r"$\tilde{p}_{t}$", 4, 0))


pgm.add_edge("h0", "FU1", linestyle="-")
pgm.add_edge("h0", "FP1", linestyle="-")
pgm.add_edge("x1", "FU1", linestyle="-")
pgm.add_edge("FU1", "h1", linestyle="-")
pgm.add_edge("FP1", "ht1", linestyle="-")
pgm.add_edge("h1", "FD1", linestyle="-")
pgm.add_edge("ht1", "FDt1", linestyle="-")
pgm.add_edge("FD1", "p1", linestyle="-")
pgm.add_edge("FDt1", "pt1", linestyle="-")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-FG-filter-predict"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


# Filter and predict two step

pgm = daft.PGM([8, 6], origin=[0, -1])

pgm.add_node(daft.Node("h0", r"$h_{t-1}$", 1, 3))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 2, 4))
pgm.add_node(daft.Node("FU1", r"$F_u$", 2.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("FP1", r"$F_p$", 2.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("FD1", r"$F_d$", 3.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("FDt1", r"$F_d$", 4.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("h1", r"$h_{t}$", 3, 3))
pgm.add_node(daft.Node("ht1", r"$\tilde{h}_{t}$", 4, 2))
pgm.add_node(daft.Node("p1", r"$p_{t}$", 3, 0))
pgm.add_node(daft.Node("pt1", r"$\tilde{p}_{t}$", 4, 0))

pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 5, 4))
pgm.add_node(daft.Node("FU2", r"$F_u$", 5.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("FP2", r"$F_p$", 5.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("FD2", r"$F_d$", 6.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("FDt2", r"$F_d$", 7.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("h2", r"$h_{t+1}$", 6, 3))
pgm.add_node(daft.Node("ht2", r"$\tilde{h}_{t+1}$", 7, 2))
pgm.add_node(daft.Node("p2", r"$p_{t+1}$", 6, 0))
pgm.add_node(daft.Node("pt2", r"$\tilde{p}_{t+1}$", 7, 0))

pgm.add_edge("h0", "FU1", linestyle="-")
pgm.add_edge("h0", "FP1", linestyle="-")
pgm.add_edge("x1", "FU1", linestyle="-")
pgm.add_edge("FU1", "h1", linestyle="-")
pgm.add_edge("FP1", "ht1", linestyle="-")
pgm.add_edge("h1", "FD1", linestyle="-")
pgm.add_edge("ht1", "FDt1", linestyle="-")
pgm.add_edge("FD1", "p1", linestyle="-")
pgm.add_edge("FDt1", "pt1", linestyle="-")

pgm.add_edge("h1", "FU2", linestyle="-")
pgm.add_edge("ht1", "FP2", linestyle="-")
pgm.add_edge("x2", "FU2", linestyle="-")
pgm.add_edge("FU2", "h2", linestyle="-")
pgm.add_edge("FP2", "ht2", linestyle="-")
pgm.add_edge("h2", "FD2", linestyle="-")
pgm.add_edge("ht2", "FDt2", linestyle="-")
pgm.add_edge("FD2", "p2", linestyle="-")
pgm.add_edge("FDt2", "pt2", linestyle="-")



pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-FG-filter-predict-twostep"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))




# Filter 

pgm = daft.PGM([5, 6], origin=[0, -1])

pgm.add_node(daft.Node("h0", r"$h_{t-1}$", 1, 3))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 2, 4))
pgm.add_node(daft.Node("FU1", r"$F_u$", 2.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("FD1", r"$F_d$", 3.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("h1", r"$h_{t}$", 3, 3))
pgm.add_node(daft.Node("p1", r"$p_{t}$", 3, 0))

if 0:
    pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 5, 4))
    pgm.add_node(daft.Node("FU2", r"$F_u$", 5.0, 3, shape="rectangle"))
    pgm.add_node(daft.Node("FD2", r"$F_d$", 6.0, 1, shape="rectangle"))
    pgm.add_node(daft.Node("h2", r"$h_{t+1}$", 6, 3))
    pgm.add_node(daft.Node("p2", r"$p_{t+1}$", 6, 0))

pgm.add_edge("h0", "FU1", linestyle="-")
pgm.add_edge("x1", "FU1", linestyle="-")
pgm.add_edge("FU1", "h1", linestyle="-")
pgm.add_edge("h1", "FD1", linestyle="-")
pgm.add_edge("FD1", "p1", linestyle="-")

if 0:
    pgm.add_edge("h1", "FU2", linestyle="-")
    pgm.add_edge("x2", "FU2", linestyle="-")
    pgm.add_edge("FU2", "h2", linestyle="-")
    pgm.add_edge("h2", "FD2", linestyle="-")
    pgm.add_edge("FD2", "p2", linestyle="-")



pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-FG-filter"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


# ProbNet

pgm = daft.PGM([5, 6], origin=[0, -1])

pgm.add_node(daft.Node("h0", r"$h_{t-1}$", 1, 3))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 2, 4))
pgm.add_node(daft.Node("FU1", r"$F_e$", 2.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("FP1", r"$F_p$", 2.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("FDt1", r"$F_d$", 4.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("h1", r"$h_{t}$", 3, 3))
pgm.add_node(daft.Node("ht1", r"$\tilde{h}_{t}$", 4, 2))
pgm.add_node(daft.Node("pt1", r"$\tilde{p}_{t}$", 4, 0))

if 0:
    pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 5, 4))
    pgm.add_node(daft.Node("FU2", r"$F_e$", 5.0, 3, shape="rectangle"))
    pgm.add_node(daft.Node("FP2", r"$F_p$", 5.0, 2, shape="rectangle"))
    pgm.add_node(daft.Node("FDt2", r"$F_d$", 7.0, 1, shape="rectangle"))
    pgm.add_node(daft.Node("h2", r"$h_{t+1}$", 6, 3))
    pgm.add_node(daft.Node("ht2", r"$\tilde{h}_{t}$", 7, 2))
    pgm.add_node(daft.Node("pt2", r"$\tilde{p}_{t+1}$", 7, 0))

pgm.add_edge("h0", "FP1", linestyle="-")
pgm.add_edge("x1", "FU1", linestyle="-")
pgm.add_edge("FU1", "h1", linestyle="-")
pgm.add_edge("FP1", "ht1", linestyle="-")
pgm.add_edge("ht1", "FDt1", linestyle="-")
pgm.add_edge("FDt1", "pt1", linestyle="-")

if 0:
    pgm.add_edge("ht1", "FP2", linestyle="-")
    pgm.add_edge("x2", "FU2", linestyle="-")
    pgm.add_edge("FU2", "h2", linestyle="-")
    pgm.add_edge("FP2", "ht2", linestyle="-")
    pgm.add_edge("ht2", "FDt2", linestyle="-")
    pgm.add_edge("FDt2", "pt2", linestyle="-")



pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-FG-probnet"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))




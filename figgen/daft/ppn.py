from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
#rc("text.latex", preamble=open("macros.tex").read())

import os

#import imp
#daft = imp.load_source('daft', 'daft-080308/daft.py')
import daft


# Two objects, discrim model

pgm = daft.PGM([10, 6], origin=[0, 0])

pgm.add_node(daft.Node("x1", r"$x_{t-1}$", 2, 5, observed=True))
pgm.add_node(daft.Node("x2", r"$x_{t}$", 5, 5, observed=True))

pgm.add_node(daft.Node("z1", r"$s_{t-1}$", 2, 4))
pgm.add_node(daft.Node("z2", r"$s_{t}$", 5, 4))
pgm.add_node(daft.Node("z3", r"$s_{t+1}$", 8, 4))

pgm.add_node(daft.Node("yA1", r"$p^1_{t-1}$", 1, 3))
pgm.add_node(daft.Node("yA2", r"$p^1_{t}$", 4, 3))
pgm.add_node(daft.Node("yA3", r"$p^1_{t+1}$", 7, 3))

pgm.add_node(daft.Node("yB1", r"$p^2_{t-1}$", 3, 2))
pgm.add_node(daft.Node("yB2", r"$p^2_{t}$", 6, 2))
pgm.add_node(daft.Node("yB3", r"$p^2_{t+1}$", 9, 2))

pgm.add_node(daft.Node("LA1", r"$L^1_{t-1}$", 1, 1))
pgm.add_node(daft.Node("LB1", r"$L^2_{t-1}$", 3, 1, observed=True))

pgm.add_node(daft.Node("LA2", r"$L^1_{t}$", 4, 1, observed=True))
pgm.add_node(daft.Node("LB2", r"$L^2_{t}$", 6, 1))

pgm.add_node(daft.Node("LA3", r"$L^1_{t+1}$", 7, 1, observed=True))
pgm.add_node(daft.Node("LB3", r"$L^2_{t+1}$", 9, 1, observed=True))

pgm.add_edge("z1", "z2", linestyle="-")
pgm.add_edge("z2", "z3", linestyle="-")
pgm.add_edge("yA1", "yA2", linestyle="-")
pgm.add_edge("yA2", "yA3", linestyle="-")
pgm.add_edge("yB1", "yB2", linestyle="-")
pgm.add_edge("yB2", "yB3", linestyle="-")

pgm.add_edge("z1", "yA1", linestyle="-")
pgm.add_edge("z2", "yA2", linestyle="-")
pgm.add_edge("z3", "yA3", linestyle="-")
pgm.add_edge("z1", "yB1", linestyle="-")
pgm.add_edge("z2", "yB2", linestyle="-")
pgm.add_edge("z3", "yB3", linestyle="-")

pgm.add_edge("x1", "z1", linestyle="-")
pgm.add_edge("x2", "z2", linestyle="-")

pgm.add_edge("yA1", "LA1", linestyle="--")
pgm.add_edge("yB1", "LB1", linestyle="-")
pgm.add_edge("yA2", "LA2", linestyle="--")
pgm.add_edge("yB2", "LB2", linestyle="-")
pgm.add_edge("yA3", "LA3", linestyle="-")
pgm.add_edge("yB3", "LB3", linestyle="-")

pgm.add_edge("yA1", "yB2", linestyle=":")
pgm.add_edge("yA2", "yB3", linestyle=":")
pgm.add_edge("yB1", "yA2", linestyle=":")
pgm.add_edge("yB2", "yA3", linestyle=":")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-two-objects-disc"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))





# Two objects, gen model

pgm = daft.PGM([10, 6], origin=[0, 0])

pgm.add_node(daft.Node("z1", r"$z_{t-1}$", 2, 4))
pgm.add_node(daft.Node("z2", r"$z_{t}$", 5, 4))
pgm.add_node(daft.Node("z3", r"$z_{t+1}$", 8, 4))

pgm.add_node(daft.Node("yA1", r"$y^1_{t-1}$", 1, 3))
pgm.add_node(daft.Node("yA2", r"$y^1_{t}$", 4, 3, observed=True))
pgm.add_node(daft.Node("yA3", r"$y^1_{t+1}$", 7, 3, observed=True))

pgm.add_node(daft.Node("yB1", r"$y^2_{t-1}$", 3, 2, observed=True))
pgm.add_node(daft.Node("yB2", r"$y^2_{t}$", 6, 2))
pgm.add_node(daft.Node("yB3", r"$y^2_{t+1}$", 9, 2, observed=True))

pgm.add_node(daft.Node("x1", r"$x_{t-1}$", 2, 1, observed=True))
pgm.add_node(daft.Node("x2", r"$x_{t}$", 5, 1, observed=True))

pgm.add_edge("z1", "z2", linestyle="-")
pgm.add_edge("z2", "z3", linestyle="-")
pgm.add_edge("yA1", "yA2", linestyle="-")
pgm.add_edge("yA2", "yA3", linestyle="-")
pgm.add_edge("yB1", "yB2", linestyle="-")
pgm.add_edge("yB2", "yB3", linestyle="-")

pgm.add_edge("z1", "yA1", linestyle="-")
pgm.add_edge("z2", "yA2", linestyle="-")
pgm.add_edge("z3", "yA3", linestyle="-")
pgm.add_edge("z1", "yB1", linestyle="-")
pgm.add_edge("z2", "yB2", linestyle="-")
pgm.add_edge("z3", "yB3", linestyle="-")

pgm.add_edge("z1", "x1", linestyle="-")
pgm.add_edge("yA1", "x1", linestyle="-")
pgm.add_edge("yB1", "x1", linestyle="-")

pgm.add_edge("z2", "x2", linestyle="-")
pgm.add_edge("yA2", "x2", linestyle="-")
pgm.add_edge("yB2", "x2", linestyle="-")

pgm.add_edge("yA1", "yB1", linestyle=":")
pgm.add_edge("yA2", "yB2", linestyle=":")
pgm.add_edge("yA3", "yB3", linestyle=":")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-two-objects-gen"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


input('done')
exit()

# Two objects, partially observed

pgm = daft.PGM([8, 6], origin=[-1, 0])

pgm.add_node(daft.Node("s1", r"$s_{t}$", 1, 4))
pgm.add_node(daft.Node("s2", r"$s_{t+1}$", 3, 4))
pgm.add_node(daft.Node("s3", r"$s_{t+2}$", 5, 4))

pgm.add_node(daft.Node("p1A", r"$p^1_{t}$", 1, 3))
pgm.add_node(daft.Node("p1B", r"$p^2_{t}$", 2, 2))
pgm.add_node(daft.Node("p2A", r"$p^1_{t+1}$", 3, 3))
pgm.add_node(daft.Node("p2B", r"$p^2_{t+1}$", 4, 2))
pgm.add_node(daft.Node("p3A", r"$p^1_{t+2}$", 5, 3))
pgm.add_node(daft.Node("p3B", r"$p^2_{t+2}$", 6, 2))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 1, 5, observed=True))
pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 3, 5, observed=True))
pgm.add_node(daft.Node("y1A", r"$L_t^1$", 1, 1))
pgm.add_node(daft.Node("y1B", r"$L_t^2$", 2, 1, observed=True))
pgm.add_node(daft.Node("y2A", r"$L_{t+1}^1$", 3, 1, observed=True))
pgm.add_node(daft.Node("y2B", r"$L_{t+1}^2$", 4, 1))
pgm.add_node(daft.Node("y3A", r"$L_{t+2}^1$", 5, 1, observed=True))
pgm.add_node(daft.Node("y3B", r"$L_{t+2}^2$", 6, 1, observed=True))

pgm.add_edge("x1", "s1", linestyle="-")
pgm.add_edge("x2", "s2", linestyle="-")

pgm.add_edge("s1", "s2", linestyle="-")
pgm.add_edge("s2", "s3", linestyle="-")

pgm.add_edge("p1A", "p2A", linestyle="-")
pgm.add_edge("p1B", "p2B", linestyle="-")
pgm.add_edge("p2A", "p3A", linestyle="-")
pgm.add_edge("p2B", "p3B", linestyle="-")

pgm.add_edge("s1", "p1A", linestyle="-")
pgm.add_edge("s1", "p1B", linestyle="-")
pgm.add_edge("s2", "p2A", linestyle="-")
pgm.add_edge("s2", "p2B", linestyle="-")
pgm.add_edge("s3", "p3A", linestyle="-")
pgm.add_edge("s3", "p3B", linestyle="-")

pgm.add_edge("p1A", "y1A", linestyle="--")
pgm.add_edge("p1B", "y1B", linestyle="-")
pgm.add_edge("p2A", "y2A", linestyle="-")
pgm.add_edge("p2B", "y2B", linestyle="--")
pgm.add_edge("p3A", "y3A", linestyle="-")
pgm.add_edge("p3B", "y3B", linestyle="-")


pgm.add_edge("p1A", "p2B", linestyle=":")
pgm.add_edge("p1B", "p2A", linestyle=":")
pgm.add_edge("p2A", "p3B", linestyle=":")
pgm.add_edge("p2B", "p3A", linestyle=":")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-two-objects-partial"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))



# Two objects

pgm = daft.PGM([6, 6], origin=[-1, 0])

pgm.add_node(daft.Node("s1", r"$s_{t}$", 1, 4))
pgm.add_node(daft.Node("s2", r"$s_{t+1}$", 3, 4))

pgm.add_node(daft.Node("p1A", r"$p^1_{t}$", 1, 3))
pgm.add_node(daft.Node("p2A", r"$p^1_{t+1}$", 3, 3))
pgm.add_node(daft.Node("p1B", r"$p^2_{t}$", 2, 2))
pgm.add_node(daft.Node("p2B", r"$p^2_{t+1}$", 4, 2))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 1, 5))
pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 3, 5))
pgm.add_node(daft.Node("y1A", r"$\hat{y}^1_{t}$", 1, 1))
pgm.add_node(daft.Node("y2A", r"$\hat{y}^1_{t+1}$", 3, 1))
pgm.add_node(daft.Node("y1B", r"$\hat{y}^2_{t}$", 2, 1))
pgm.add_node(daft.Node("y2B", r"$\hat{y}^2_{t+1}$", 4, 1))

pgm.add_edge("x1", "s1", linestyle="-")
pgm.add_edge("x2", "s2", linestyle="-")
pgm.add_edge("p1A", "y1A", linestyle="-")
pgm.add_edge("p2A", "y2A", linestyle="-")
pgm.add_edge("s1", "s2", linestyle="-")
pgm.add_edge("p1A", "p2A", linestyle="-")
pgm.add_edge("p1B", "p2B", linestyle="-")
pgm.add_edge("s1", "p1A", linestyle="-")
pgm.add_edge("s2", "p2A", linestyle="-")
pgm.add_edge("s1", "p1B", linestyle="-")
pgm.add_edge("s2", "p2B", linestyle="-")
pgm.add_edge("p1B", "y1B", linestyle="-")
pgm.add_edge("p2B", "y2B", linestyle="-")
pgm.add_edge("p1A", "p2B", linestyle="-")
pgm.add_edge("p1B", "p2A", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-two-objects"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))



# Three objecsts

pgm = daft.PGM([8, 8], origin=[-1, 0])

pgm.add_node(daft.Node("s1", r"$s_{t}$", 1, 5))
pgm.add_node(daft.Node("s2", r"$s_{t+1}$", 4, 5))

pgm.add_node(daft.Node("p1A", r"$p^A_{t}$", 1, 4))
pgm.add_node(daft.Node("p1B", r"$p^B_{t}$", 2, 3))
pgm.add_node(daft.Node("p1C", r"$p^C_{t}$", 3, 2))

pgm.add_node(daft.Node("p2A", r"$p^A_{t+1}$", 4, 4))
pgm.add_node(daft.Node("p2B", r"$p^B_{t+1}$", 5, 3))
pgm.add_node(daft.Node("p2C", r"$p^C_{t+1}$", 6, 2))

pgm.add_node(daft.Node("x1", r"$x_{t}$", 1, 6))
pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 4, 6))


pgm.add_edge("x1", "s1", linestyle="-")
pgm.add_edge("x2", "s2", linestyle="-")

pgm.add_edge("s1", "p1A", linestyle="-")
pgm.add_edge("s1", "p1B", linestyle="-")
pgm.add_edge("s1", "p1C", linestyle="-")
pgm.add_edge("s2", "p2A", linestyle="-")
pgm.add_edge("s2", "p2B", linestyle="-")
pgm.add_edge("s2", "p2C", linestyle="-")

pgm.add_edge("s1", "s2", linestyle="-")
pgm.add_edge("p1A", "p2A", linestyle="-")
pgm.add_edge("p1B", "p2B", linestyle="-")
pgm.add_edge("p1C", "p2C", linestyle="-")

pgm.add_edge("p1A", "p2B", linestyle="-")
pgm.add_edge("p1B", "p2A", linestyle="-")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-three-objects"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))




# Predict one step

pgm = daft.PGM([5, 7], origin=[-1, -1])

pgm.add_node(daft.Node("h0", r"$h_{t-1}$", 0, 4))
pgm.add_node(daft.Node("h1", r"$h_{t}$", 1, 4))
pgm.add_node(daft.Node("h2", r"$h_{t+1}$", 2, 4))
pgm.add_node(daft.Node("ht1", r"$\tilde{h}_{t}$", 1, 2))
pgm.add_node(daft.Node("ht2", r"$\tilde{h}_{t+1}$", 2, 2))
pgm.add_node(daft.Node("x1", r"$x_{t}$", 1, 5))
pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 2, 5))
pgm.add_node(daft.Node("y1", r"$\hat{y}_{t}$", 1, 3))
pgm.add_node(daft.Node("y2", r"$\hat{y}_{t+1}$", 2, 3))
pgm.add_node(daft.Node("yt1", r"$\tilde{y}_{t}$", 1, 1))
pgm.add_node(daft.Node("yt2", r"$\tilde{y}_{t+1}$", 2, 1))

pgm.add_edge("h0", "h1", linestyle="-")
pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h0", "ht1", linestyle="-")
pgm.add_edge("h1", "ht2", linestyle="-")
pgm.add_edge("x1", "h1", linestyle="-")
pgm.add_edge("x2", "h2", linestyle="-")
pgm.add_edge("h1", "y1", linestyle="-")
pgm.add_edge("h2", "y2", linestyle="-")
pgm.add_edge("ht1", "yt1", linestyle="-")
pgm.add_edge("ht2", "yt2", linestyle="-")
pgm.add_edge("y1", "h2", linestyle=":")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-predict-one-step"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))





# Predict one step structured

pgm = daft.PGM([4, 8], origin=[-1, 0])

pgm.add_node(daft.Node("s0", r"$s_{t-1}$", 0, 6))
pgm.add_node(daft.Node("p0", r"$p_{t-1}$", 0, 5))
pgm.add_node(daft.Node("s1", r"$s_{t}$", 1, 6))
pgm.add_node(daft.Node("p1", r"$p_{t}$", 1, 5))
pgm.add_node(daft.Node("s2", r"$s_{t+1}$", 2, 6))
pgm.add_node(daft.Node("p2", r"$p_{t+1}$", 2, 5))
pgm.add_node(daft.Node("st1", r"$\tilde{s}_{t}$", 1, 3))
pgm.add_node(daft.Node("pt1", r"$\tilde{p}_{t}$", 1, 2))
pgm.add_node(daft.Node("st2", r"$\tilde{s}_{t+1}$", 2, 3))
pgm.add_node(daft.Node("pt2", r"$\tilde{p}_{t+1}$", 2, 2))


pgm.add_node(daft.Node("x1", r"$x_{t}$", 1, 7))
pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 2, 7))
pgm.add_node(daft.Node("y1", r"$\hat{y}_{t}$", 1, 4))
pgm.add_node(daft.Node("y2", r"$\hat{y}_{t+1}$", 2, 4))
pgm.add_node(daft.Node("yt1", r"$\tilde{y}_{t}$", 1, 1))
pgm.add_node(daft.Node("yt2", r"$\tilde{y}_{t+1}$", 2, 1))

pgm.add_edge("x1", "s1", linestyle="-")
pgm.add_edge("x2", "s2", linestyle="-")
pgm.add_edge("p1", "y1", linestyle="-")
pgm.add_edge("p2", "y2", linestyle="-")
pgm.add_edge("pt1", "yt1", linestyle="-")
pgm.add_edge("pt2", "yt2", linestyle="-")
pgm.add_edge("s0", "s1", linestyle="-")
pgm.add_edge("p0", "p1", linestyle="-")
pgm.add_edge("s1", "s2", linestyle="-")
pgm.add_edge("p1", "p2", linestyle="-")
pgm.add_edge("s0", "st1", linestyle="-")
pgm.add_edge("p0", "pt1", linestyle="-")
pgm.add_edge("s1", "st2", linestyle="-")
pgm.add_edge("p1", "pt2", linestyle="-")
pgm.add_edge("s1", "p1", linestyle="-")
pgm.add_edge("s2", "p2", linestyle="-")
pgm.add_edge("st1", "pt1", linestyle="-")
pgm.add_edge("st2", "pt2", linestyle="-")
pgm.add_edge("y1", "s2", linestyle=":")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-factored"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


# Predict twp step

pgm = daft.PGM([6, 8], origin=[-1, -2])

pgm.add_node(daft.Node("h0", r"$h_{t-1}$", 0, 4))
pgm.add_node(daft.Node("h1", r"$h_{t}$", 1, 4))
pgm.add_node(daft.Node("h2", r"$h_{t+1}$", 2, 4))
pgm.add_node(daft.Node("x1", r"$x_{t}$", 1, 5))
pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 2, 5))
pgm.add_node(daft.Node("y1", r"$\hat{y}_{t}$", 1, 3))
pgm.add_node(daft.Node("y2", r"$\hat{y}_{t+1}$", 2, 3))
pgm.add_node(daft.Node("yt1", r"$\tilde{y}_{t}^{t-1}$", 1, 1))
pgm.add_node(daft.Node("yt1B", r"$\tilde{y}_{t+1}^{t-1}$", 2, 1))
pgm.add_node(daft.Node("yt2", r"$\tilde{y}_{t+1}^{t}$", 2, -1))
pgm.add_node(daft.Node("yt2B", r"$\tilde{y}_{t+2}^{t}$", 3, -1))

pgm.add_node(daft.Node("ht1", r"$\tilde{h}_{t}^{t-1}$", 1, 2))
pgm.add_node(daft.Node("ht1B", r"$\tilde{h}_{t+1}^{t-1}$", 2, 2))
pgm.add_node(daft.Node("ht2", r"$\tilde{h}_{t+1}^{t}$", 2, 0))
pgm.add_node(daft.Node("ht2B", r"$\tilde{h}_{t+2}^{t}$", 3, 0))


pgm.add_edge("h0", "h1", linestyle="-")
pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h0", "ht1", linestyle="-")
pgm.add_edge("h1", "ht2", linestyle="-")
pgm.add_edge("ht1", "ht1B", linestyle="-")
pgm.add_edge("ht2", "ht2B", linestyle="-")
pgm.add_edge("x1", "h1", linestyle="-")
pgm.add_edge("x2", "h2", linestyle="-")
pgm.add_edge("h1", "y1", linestyle="-")
pgm.add_edge("h2", "y2", linestyle="-")
pgm.add_edge("ht1", "yt1", linestyle="-")
pgm.add_edge("ht1B", "yt1B", linestyle="-")
pgm.add_edge("ht2", "yt2", linestyle="-")
pgm.add_edge("ht2B", "yt2B", linestyle="-")
pgm.add_edge("yt1", "ht1B", linestyle=":")
pgm.add_edge("yt2", "ht2B", linestyle=":")
pgm.add_edge("y1", "h2", linestyle=":")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-predict-two-step"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))


# Cond RNN 

pgm = daft.PGM([4, 4], origin=[-1, 2])

pgm.add_node(daft.Node("h0", r"$h_{t-1}$", 0, 4))
pgm.add_node(daft.Node("h1", r"$h_{t}$", 1, 4))
pgm.add_node(daft.Node("h2", r"$h_{t+1}$", 2, 4))
pgm.add_node(daft.Node("x1", r"$x_{t}$", 1, 5))
pgm.add_node(daft.Node("x2", r"$x_{t+1}$", 2, 5))
pgm.add_node(daft.Node("y1", r"$\hat{y}_{t}$", 1, 3))
pgm.add_node(daft.Node("y2", r"$\hat{y}_{t+1}$", 2, 3))

pgm.add_edge("h0", "h1", linestyle="-")
pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("x1", "h1", linestyle="-")
pgm.add_edge("x2", "h2", linestyle="-")
pgm.add_edge("h1", "y1", linestyle="-")
pgm.add_edge("h2", "y2", linestyle="-")
pgm.add_edge("y1", "h2", linestyle=":")


pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-cond-RNN"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))



# Joint RNN 

pgm = daft.PGM([4, 4], origin=[-1, 2])

pgm.add_node(daft.Node("h0", r"$h_{t-1}$", 0, 4))
pgm.add_node(daft.Node("h1", r"$h_{t}$", 1, 4))
pgm.add_node(daft.Node("h2", r"$h_{t+1}$", 2, 4))
pgm.add_node(daft.Node("x1", r"$\hat{x}_{t}$", 1, 5))
pgm.add_node(daft.Node("x2", r"$\hat{x}_{t+1}$", 2, 5))
pgm.add_node(daft.Node("y1", r"$\hat{y}_{t}$", 1, 3))
pgm.add_node(daft.Node("y2", r"$\hat{y}_{t+1}$", 2, 3))

pgm.add_edge("h0", "h1", linestyle="-")
pgm.add_edge("h1", "h2", linestyle="-")
pgm.add_edge("h1", "x1",  linestyle="-")
pgm.add_edge("h2", "x2", linestyle="-")
pgm.add_edge("h1", "y1", linestyle="-")
pgm.add_edge("h2", "y2", linestyle="-")
pgm.add_edge("y1", "h2", linestyle=":")
pgm.add_edge("x1", "h2", linestyle=":")

pgm.render()
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "PPN-joint-RNN"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))




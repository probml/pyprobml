# Object oriented Generative image model
# oo-gen-obs
# oo-gen-dynamics
# oo-inf-loc
# oo-inf-state

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

#import daft
import os


import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')
import daft

#####



pgm = daft.PGM([7, 4], origin=[0, 0], node_unit=1.5)
pgm.add_node(daft.Node("L", r"$L_t$", 1, 3))
pgm.add_node(daft.Node("A0", r"$A_{t,0}$", 1, 2))
pgm.add_node(daft.Node("Ak", r"$A_{t,1:K}$", 1, 1))
pgm.add_node(daft.Node("Ihat0", r"$\hat{I}_{t,0}$", 3, 2))

pgm.add_node(daft.Node("slice", r"slice", 4.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("decode0", r"decode", 2.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("decodek", r"decode", 2.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("CC", r"CC", 4.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("combine", r"+", 5.0, 2, shape="rectangle"))

pgm.add_node(daft.Node("Sk", r"$\tilde{I}_{t,1:K}$", 3, 1))
pgm.add_node(daft.Node("Ihatk", r"$\hat{I}_{t,1:K}$", 5, 1))
pgm.add_node(daft.Node("I", r"$I_{t}$", 6, 2))

pgm.add_edge("L", "slice", linestyle="-")
pgm.add_edge("A0", "decode0", linestyle="-")
pgm.add_edge("Ak", "decodek", linestyle="-")
pgm.add_edge("decodek", "Sk", linestyle="-")
pgm.add_edge("Sk", "CC", linestyle="-")
pgm.add_edge("slice", "CC", linestyle="-")
pgm.add_edge("CC", "Ihatk", linestyle="-")
pgm.add_edge("Ihatk", "combine", linestyle="-")
pgm.add_edge("Ihat0", "combine", linestyle="-")
pgm.add_edge("decode0", "Ihat0", linestyle="-")
pgm.add_edge("combine", "I", linestyle="-")

pgm.render()
fname = "oo-gen-obs"
folder = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


#input('done')
#exit()



####

pgm = daft.PGM([5, 4], origin=[0, 0], node_unit=1.5)

pgm.add_node(daft.Node("L", r"$L_{t}$", 1, 3))
pgm.add_node(daft.Node("I", r"$I_{t}$", 1, 2))
pgm.add_node(daft.Node("Aold", r"$A_{t-1,0:K}$", 1, 1))
pgm.add_node(daft.Node("Anew", r"$A_{t,0:K}$", 4, 1))
pgm.add_node(daft.Node("Ik", r"$\overset{\smile}{I}_{t,0:K}$", 3, 2))
pgm.add_node(daft.Node("loc", r"slice", 2.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("attn", r"attn", 2.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("upA", r"updateA", 3.0, 1, shape="rectangle"))

pgm.add_edge("L", "loc", linestyle="-")
pgm.add_edge("I", "attn", linestyle="-")
pgm.add_edge("loc", "attn", linestyle="-")
pgm.add_edge("attn", "Ik", linestyle="-")
pgm.add_edge("Aold", "upA", linestyle="-")
pgm.add_edge("upA", "Anew", linestyle="-")
pgm.add_edge("Ik", "upA", linestyle="-")

pgm.render()
fname = "oo-inf-app"
folder = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))




############

pgm = daft.PGM([8, 5], origin=[0, -1], node_unit=1.5)

pgm.add_node(daft.Node("It", r"$I_{t}$", 4, 0))

pgm.add_node(daft.Node("Lold", r"$L_{t-1}$", 1, 3))
pgm.add_node(daft.Node("Lnew", r"$L_{t}$", 7, 3))
pgm.add_node(daft.Node("upL", r"updateL", 6.0, 3, shape="rectangle"))

pgm.add_node(daft.Node("Aold0", r"$A_{t-1,0}$", 1, 2))
pgm.add_node(daft.Node("decode0", r"decode", 2.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("I0", r"$\hat{I}_{t-1,0}$", 3, 2))
pgm.add_node(daft.Node("subtract", r"-", 4.0, 2, shape="rectangle"))
pgm.add_node(daft.Node("delta", r"$\Delta_{t,0}$", 5, 2))

pgm.add_node(daft.Node("Aoldk", r"$A_{t-1,1:K}$", 1, 1))
pgm.add_node(daft.Node("decodek", r"decode", 2.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("Ik", r"$\tilde{I}_{t-1,k}$", 3, 1))
pgm.add_node(daft.Node("CCF", r"CCF", 5.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("Hk", r"$H_{t,k}$", 6, 1))


pgm.add_edge("Aold0", "decode0", linestyle="-")
pgm.add_edge("decode0", "I0", linestyle="-")
pgm.add_edge("I0", "subtract", linestyle="-")
pgm.add_edge("subtract", "delta", linestyle="-")

pgm.add_edge("Aoldk", "decodek", linestyle="-")
pgm.add_edge("decodek", "Ik", linestyle="-")
pgm.add_edge("Ik", "CCF", linestyle="-")
pgm.add_edge("CCF", "Hk", linestyle="-")

pgm.add_edge("It", "subtract", linestyle="-")
pgm.add_edge("It", "CCF", linestyle="-")
pgm.add_edge("delta", "upL", linestyle="-")
pgm.add_edge("Hk", "upL", linestyle="-")
pgm.add_edge("Lold", "upL", linestyle="-")
pgm.add_edge("upL", "Lnew", linestyle="-")


pgm.render()
fname = "oo-inf-loc"
folder = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))



#####

############

pgm = daft.PGM([4, 5], origin=[0, 0], node_unit=1.5)

pgm.add_node(daft.Node("Lold", r"$L_{t-1}$", 1, 3))
pgm.add_node(daft.Node("Aold", r"$A_{t-1,0:K}$", 1, 1))
pgm.add_node(daft.Node("ZL", r"$Z_{t,L}$", 2, 4))
pgm.add_node(daft.Node("ZA", r"$Z_{t,0:K}$", 2, 2))
pgm.add_node(daft.Node("predL", r"predL", 2.0, 3, shape="rectangle"))
pgm.add_node(daft.Node("predA", r"predA", 2.0, 1, shape="rectangle"))
pgm.add_node(daft.Node("Lnew", r"$L_{t}$", 3, 3))
pgm.add_node(daft.Node("Anew", r"$A_{t,0:K}$", 3, 1))


pgm.add_edge("Lold", "ZL", linestyle="-")
pgm.add_edge("Lold", "predL", linestyle="-")
pgm.add_edge("ZL", "predL", linestyle="-")
pgm.add_edge("predL", "Lnew", linestyle="-")

pgm.add_edge("Aold", "ZA", linestyle="-")
pgm.add_edge("Aold", "predA", linestyle="-")
pgm.add_edge("ZA", "predA", linestyle="-")
pgm.add_edge("predA", "Anew", linestyle="-")

pgm.render()
fname = "oo-gen-dyn"
folder = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))




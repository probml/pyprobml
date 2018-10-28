# Variational neural EM

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex", preamble=open("macros.tex").read())

#import daft
import os


import imp
daft = imp.load_source('daft', 'daft-080308/daft.py')
import daft


pgm = daft.PGM([10, 10], origin=[-1, 0], node_unit=1.5)

pgm.add_node(daft.Node("Aold", r"$A_{t-1}$", 0, 2))
pgm.add_node(daft.Node("Lold", r"$L_{t-1}$", 0, 6))

pgm.add_node(daft.Node("AAcur", r"$\hat{A}_{t}$", 2, 2))
pgm.add_node(daft.Node("QAcur", r"$p_{t}^A$", 2, 3))
pgm.add_node(daft.Node("ZAcur", r"$\tilde{z}_{t}^A$", 2, 4))
pgm.add_node(daft.Node("XcurPred", r"$\tilde{x}_{t}$", 2, 9))
pgm.add_node(daft.Node("Xcur", r"$x_{t}$", 3, 1))
pgm.add_node(daft.Node("LLcur", r"$\hat{L}_{t}$", 3, 6))
pgm.add_node(daft.Node("QLcur", r"$p_{t}^L$", 3, 7))
pgm.add_node(daft.Node("ZLcur", r"$\tilde{z}_{t}^L$", 3, 8))
pgm.add_node(daft.Node("Acur", r"$A_{t}$", 4, 2))
pgm.add_node(daft.Node("Lcur", r"$L_{t}$", 4, 6))

pgm.add_node(daft.Node("AAnext", r"$\hat{A}_{t+1}$", 6, 2))
pgm.add_node(daft.Node("QAnext", r"$p_{t+1}^A$", 6, 3))
pgm.add_node(daft.Node("ZAnext", r"$\tilde{z}_{t+1}^A$", 6, 4))
pgm.add_node(daft.Node("XnextPred", r"$\tilde{x}_{t+1}$", 6, 9))
pgm.add_node(daft.Node("Xnext", r"$x_{t+1}$", 7, 1))
pgm.add_node(daft.Node("LLnext", r"$\hat{L}_{t+1}$", 7, 6))
pgm.add_node(daft.Node("QLnext", r"$p_{t+1}^L$", 7, 7))
pgm.add_node(daft.Node("ZLnext", r"$\tilde{z}_{t+1}^L$", 7, 8))
pgm.add_node(daft.Node("Anext", r"$A_{t+1}$", 8, 2))
pgm.add_node(daft.Node("Lnext", r"$L_{t+1}$", 8, 6))


pgm.add_edge("Aold", "AAcur", linestyle="-")
pgm.add_edge("AAcur", "Acur", linestyle="-")
pgm.add_edge("Lold", "LLcur", linestyle="-")
pgm.add_edge("LLcur", "Lcur", linestyle="-")

pgm.add_edge("Xcur", "AAcur", linestyle="--")
pgm.add_edge("Xcur", "LLcur", linestyle="--")
pgm.add_edge("AAcur", "QAcur", linestyle="-")
pgm.add_edge("QAcur", "ZAcur", linestyle="-")
pgm.add_edge("LLcur", "QLcur", linestyle="-")
pgm.add_edge("QLcur", "ZLcur", linestyle="-")
pgm.add_edge("ZLcur", "XcurPred", linestyle="-")
pgm.add_edge("ZLcur", "Lcur", linestyle="-")
pgm.add_edge("ZAcur", "Acur", linestyle="-")
pgm.add_edge("ZAcur", "XcurPred",linestyle="-")

pgm.add_edge("Acur", "AAnext", linestyle="-")
pgm.add_edge("AAnext", "Anext", linestyle="-")
pgm.add_edge("Lcur", "LLnext", linestyle="-")
pgm.add_edge("LLnext", "Lnext", linestyle="-")

pgm.add_edge("Xnext", "AAnext", linestyle="--")
pgm.add_edge("Xnext", "LLnext", linestyle="--")
pgm.add_edge("AAnext", "QAnext", linestyle="-")
pgm.add_edge("QAnext", "ZAnext", linestyle="-")
pgm.add_edge("ZAnext", "XnextPred", linestyle="-")
pgm.add_edge("LLnext", "QLnext", linestyle="-")
pgm.add_edge("QLnext", "ZLnext", linestyle="-")
pgm.add_edge("ZLnext", "XnextPred", linestyle="-")
pgm.add_edge("ZLnext", "Lnext", linestyle="-")
pgm.add_edge("ZAnext", "Anext", linestyle="-")


pgm.render()
fname = "vnem2"
fcurer = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(fcurer, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


#input('done')
#exit()



###

pgm = daft.PGM([10, 6], origin=[-1, 0], node_unit=1.5)

pgm.add_node(daft.Node("Aold", r"$A_{t-1}$", 0, 2))

pgm.add_node(daft.Node("AAcur", r"$\hat{A}_{t}$", 2, 2))
pgm.add_node(daft.Node("QAcur", r"$p_{t}^A$", 2, 3))
pgm.add_node(daft.Node("ZAcur", r"$\tilde{z}_{t}^A$", 2, 4))
pgm.add_node(daft.Node("XcurPred", r"$\tilde{x}_{t}$", 2, 5))
pgm.add_node(daft.Node("Xcur", r"$x_{t}$", 3, 1))
pgm.add_node(daft.Node("Acur", r"$A_{t}$", 4, 2))

pgm.add_node(daft.Node("AAnext", r"$\hat{A}_{t+1}$", 6, 2))
pgm.add_node(daft.Node("QAnext", r"$p_{t+1}^A$", 6, 3))
pgm.add_node(daft.Node("ZAnext", r"$\tilde{z}_{t+1}^A$", 6, 4))
pgm.add_node(daft.Node("XnextPred", r"$\tilde{x}_{t+1}$", 6, 5))
pgm.add_node(daft.Node("Xnext", r"$x_{t+1}$", 7, 1))
pgm.add_node(daft.Node("Anext", r"$A_{t+1}$", 8, 2))


pgm.add_edge("Aold", "AAcur", linestyle="-")
pgm.add_edge("AAcur", "Acur", linestyle="-")

pgm.add_edge("Xcur", "AAcur", linestyle="--")
pgm.add_edge("AAcur", "QAcur", linestyle="-")
pgm.add_edge("QAcur", "ZAcur", linestyle="-")
pgm.add_edge("ZAcur", "Acur", linestyle="-")
pgm.add_edge("ZAcur", "XcurPred",linestyle="-")

pgm.add_edge("Acur", "AAnext", linestyle="-")

pgm.add_edge("AAnext", "Anext", linestyle="-")
pgm.add_edge("Xnext", "AAnext", linestyle="--")
pgm.add_edge("AAnext", "QAnext", linestyle="-")
pgm.add_edge("QAnext", "ZAnext", linestyle="-")
pgm.add_edge("ZAnext", "XnextPred", linestyle="-")
pgm.add_edge("ZAnext", "Anext", linestyle="-")


pgm.render()
fname = "vnem2-noL"
fcurer = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(fcurer, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


#input('done')
#exit()



###

pgm = daft.PGM([10, 4], origin=[-1, 0], node_unit=1.5)

pgm.add_node(daft.Node("AAold", r"$z^A_{t-1}$", 0, 2))

pgm.add_node(daft.Node("AAcur", r"$z^A_{t}$", 2, 2))
pgm.add_node(daft.Node("XcurPred", r"$\tilde{x}_{t}$", 2, 3))
pgm.add_node(daft.Node("Xcur", r"$x_{t}$", 3, 1))

pgm.add_node(daft.Node("AAnext", r"$z^{A}_{t+1}$", 6, 2))
pgm.add_node(daft.Node("XnextPred", r"$\tilde{x}_{t+1}$", 6, 3))
pgm.add_node(daft.Node("Xnext", r"$x_{t+1}$", 7, 1))


pgm.add_edge("AAold", "AAcur", linestyle="-")

pgm.add_edge("Xcur", "AAcur", linestyle="--")
pgm.add_edge("AAcur", "XcurPred",linestyle="-")

pgm.add_edge("AAcur", "AAnext", linestyle="-")

pgm.add_edge("Xnext", "AAnext", linestyle="--")
pgm.add_edge("AAnext", "XnextPred", linestyle="-")


pgm.render()
fname = "vnem2-markov"
fcurer = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(fcurer, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


#input('done')
#exit()


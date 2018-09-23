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


pgm = daft.PGM([9, 7], origin=[-1, 0], node_unit=1.5)

pgm.add_node(daft.Node("Aold", r"$H_{t-1}^I$", 0, 2))
pgm.add_node(daft.Node("XAold", r"$\tilde{x}_{t-1}$", 0, 5))
pgm.add_node(daft.Node("ZAold", r"$\tilde{z}_{t-1}$", 0, 4))
pgm.add_node(daft.Node("QAold", r"$q_{t-1}$", 0, 3))

pgm.add_node(daft.Node("AAcur", r"$\hat{H}_{t}^I$", 2, 2))
pgm.add_node(daft.Node("QAcur", r"$q_{t}$", 2, 3))
pgm.add_node(daft.Node("ZAcur", r"$\tilde{z}_{t}$", 2, 4))
pgm.add_node(daft.Node("XAcur", r"$\tilde{x}_{t}$", 2, 5))
pgm.add_node(daft.Node("Xcur", r"$x_{t}$", 2, 1))
pgm.add_node(daft.Node("Acur", r"$H_{t}^I$", 3, 2))
pgm.add_node(daft.Node("AAnext", r"$\hat{H}_{t+1}^G$", 5, 2))
pgm.add_node(daft.Node("QAnext", r"$p_{t+1}$", 5, 3))
pgm.add_node(daft.Node("ZAnext", r"$\tilde{z}_{t+1}$", 5, 4))
pgm.add_node(daft.Node("XAnext", r"$\tilde{x}_{t+1}$", 5, 5))
pgm.add_node(daft.Node("Anext", r"$H_{t+1}^G$", 6, 2))

pgm.add_edge("Aold", "AAcur", linestyle="-")
#pgm.add_edge("XAold", "AAcur", linestyle="-")
#pgm.add_edge("ZAold", "AAcur", linestyle="-")

pgm.add_edge("Xcur", "AAcur", linestyle="-")
pgm.add_edge("AAcur", "Acur", linestyle="-")
pgm.add_edge("AAcur", "QAcur", linestyle="-")
pgm.add_edge("QAcur", "ZAcur", linestyle="-")
pgm.add_edge("ZAcur", "XAcur", linestyle="-")
pgm.add_edge("ZAcur", "Acur", linestyle="-")

pgm.add_edge("Acur", "AAnext", linestyle="-")
pgm.add_edge("AAnext", "Anext", linestyle="-")
#pgm.add_edge("XAcur", "AAnext", linestyle="-")
pgm.add_edge("AAnext", "QAnext", linestyle="-")
pgm.add_edge("QAnext", "ZAnext", linestyle="-")
pgm.add_edge("ZAnext", "XAnext", linestyle="-")
pgm.add_edge("ZAnext", "Anext", linestyle="-")


pgm.render()
fname = "vnem-svglp"
folder = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


#input('done')
#exit()

pgm = daft.PGM([8, 10], origin=[-1, 0], node_unit=1.5)
greek = False

if greek:
    pgm.add_node(daft.Node("Aold", r"$\theta_{t-1}$", 0, 2))
    pgm.add_node(daft.Node("XAold", r"$\psi_{t-1}$", 0, 5))
    pgm.add_node(daft.Node("QLold", r"$\gamma_{t-1}$", 0, 7))
    
    pgm.add_node(daft.Node("AAcur", r"$\theta_{t}$", 2, 2))
    pgm.add_node(daft.Node("XAcur", r"$\psi_{t}$", 2, 5))
    pgm.add_node(daft.Node("XcurPred", r"$\tilde{x}_{t}$", 2, 9))
    pgm.add_node(daft.Node("Xcur", r"$x_{t}$", 3, 1))
    pgm.add_node(daft.Node("QLcur", r"$\gamma_{t}$", 3, 7))
    
    pgm.add_node(daft.Node("AAnext", r"$\theta_{t+1}$", 5, 2))
    pgm.add_node(daft.Node("XAnext", r"$\psi_{t+1}$", 5, 5))
    pgm.add_node(daft.Node("XnextPred", r"$\tilde{x}_{t+1}$", 5, 9))
    pgm.add_node(daft.Node("Xnext", r"$\tilde{x}_{t}$", 6, 1))
    pgm.add_node(daft.Node("QLnext", r"$\gamma_{t+1}$", 6, 7))
else:
    pgm.add_node(daft.Node("Aold", r"$A_{t-1}$", 0, 2))
    pgm.add_node(daft.Node("XAold", r"$z^A_{t-1}$", 0, 5))
    pgm.add_node(daft.Node("QLold", r"$q^L_{t-1}$", 0, 7))
    
    pgm.add_node(daft.Node("AAcur", r"$A_{t}$", 2, 2))
    pgm.add_node(daft.Node("XAcur", r"$z^A_{t}$", 2, 5))
    pgm.add_node(daft.Node("XcurPred", r"$\tilde{x}_{t}$", 2, 9))
    pgm.add_node(daft.Node("Xcur", r"$x_{t}$", 3, 1))
    pgm.add_node(daft.Node("QLcur", r"$q^L_{t}$", 3, 7))
    
    pgm.add_node(daft.Node("AAnext", r"$A_{t+1}$", 5, 2))
    pgm.add_node(daft.Node("XAnext", r"$z^A_{t+1}$", 5, 5))
    pgm.add_node(daft.Node("XnextPred", r"$\tilde{x}_{t+1}$", 5, 9))
    pgm.add_node(daft.Node("Xnext", r"$\tilde{x}_{t}$", 6, 1))
    pgm.add_node(daft.Node("QLnext", r"$q^L_{t+1}$", 6, 7))



pgm.add_edge("Aold", "AAcur", linestyle="-")
pgm.add_edge("XAold", "AAcur", linestyle="-")
pgm.add_edge("QLold", "AAcur", linestyle="-")
pgm.add_edge("Xcur", "AAcur", linestyle="-")
pgm.add_edge("Xcur", "QLcur", linestyle="-")
pgm.add_edge("AAcur", "XAcur", linestyle="-")
pgm.add_edge("XAcur", "XcurPred", linestyle="-")
pgm.add_edge("QLcur", "XcurPred", linestyle="-")
pgm.add_edge("XAcur", "QLcur", linestyle="-")


pgm.add_edge("AAcur", "AAnext", linestyle="-")
pgm.add_edge("XAcur", "AAnext", linestyle="-")
pgm.add_edge("QLcur", "AAnext", linestyle="-")
pgm.add_edge("Xnext", "AAnext", linestyle="-")
pgm.add_edge("Xnext", "QLnext", linestyle="-")
pgm.add_edge("AAnext", "XAnext", linestyle="-")
pgm.add_edge("XAnext", "XnextPred", linestyle="-")
pgm.add_edge("QLnext", "XnextPred", linestyle="-")
pgm.add_edge("XAnext", "QLnext", linestyle="-")



pgm.render()
if greek:
    fname = "vnem-neural-em-greek"
else:
    fname = "vnem-neural-em-nogreek"
fcurer = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(fcurer, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


input('done')
exit()


###

pgm = daft.PGM([9, 10], origin=[-1, 0], node_unit=1.5)

pgm.add_node(daft.Node("Aold", r"$A_{t-1}^I$", 0, 2))
pgm.add_node(daft.Node("XAold", r"$x^A_{t-1}$", 0, 5))
pgm.add_node(daft.Node("Lold", r"$L_{t-1}^I$", 0, 6))
pgm.add_node(daft.Node("QLold", r"$q_{t-1}^L$", 0, 7))


pgm.add_node(daft.Node("AAcur", r"$\hat{A}_{t}^I$", 2, 2))
pgm.add_node(daft.Node("QAcur", r"$q_{t}^A$", 2, 3))
pgm.add_node(daft.Node("ZAcur", r"$\tilde{z}_{t}^A$", 2, 4))
pgm.add_node(daft.Node("XAcur", r"$x_{t}^A$", 2, 5))
pgm.add_node(daft.Node("XcurPred", r"$\tilde{x}_{t}$", 2, 9))

pgm.add_node(daft.Node("Xcur", r"$x_{t}$", 3, 1))
pgm.add_node(daft.Node("LLcur", r"$\hat{L}_{t}^I$", 3, 6))
pgm.add_node(daft.Node("QLcur", r"$q_{t}^L$", 3, 7))
pgm.add_node(daft.Node("ZLcur", r"$\tilde{z}_{t}^L$", 3, 8))

pgm.add_node(daft.Node("Acur", r"$A_{t}^I$", 4, 2))
pgm.add_node(daft.Node("Lcur", r"$L_{t}^I$", 4, 6))


pgm.add_node(daft.Node("AAnext", r"$\hat{A}_{t+1}^G$", 5, 2))
pgm.add_node(daft.Node("QAnext", r"$p_{t+1}^A$", 5, 3))
pgm.add_node(daft.Node("ZAnext", r"$\tilde{z}_{t+1}^A$", 5, 4))
pgm.add_node(daft.Node("XAnext", r"$x_{t+1}^A$", 5, 5))
pgm.add_node(daft.Node("XnextPred", r"$\tilde{x}_{t+1}$", 5, 9))

pgm.add_node(daft.Node("LLnext", r"$\hat{L}_{t+1}^G$", 6, 6))
pgm.add_node(daft.Node("QLnext", r"$p_{t+1}^L$", 6, 7))
pgm.add_node(daft.Node("ZLnext", r"$\tilde{z}_{t+1}^L$", 6, 8))

pgm.add_node(daft.Node("Anext", r"$A_{t+1}^G$", 7, 2))
pgm.add_node(daft.Node("Lnext", r"$L_{t+1}^G$", 7, 6))

pgm.add_edge("Xcur", "AAcur", linestyle="-")
pgm.add_edge("Xcur", "LLcur", linestyle="-")

pgm.add_edge("Aold", "AAcur", linestyle="-")
pgm.add_edge("AAcur", "Acur", linestyle="-")
pgm.add_edge("Lold", "LLcur", linestyle="-")
pgm.add_edge("LLcur", "Lcur", linestyle="-")

pgm.add_edge("XAold", "AAcur", linestyle="-")
pgm.add_edge("QLold", "AAcur", linestyle="-")

pgm.add_edge("AAcur", "QAcur", linestyle="-")
pgm.add_edge("QAcur", "ZAcur", linestyle="-")
pgm.add_edge("ZAcur", "XAcur", linestyle="-")
pgm.add_edge("XAcur", "XcurPred", linestyle="-")
pgm.add_edge("XAcur", "LLcur", linestyle="-")
pgm.add_edge("LLcur", "QLcur", linestyle="-")
pgm.add_edge("QLcur", "ZLcur", linestyle="-")
pgm.add_edge("ZLcur", "XcurPred", linestyle="-")

pgm.add_edge("ZLcur", "Lcur", linestyle="-")
pgm.add_edge("ZAcur", "Acur", linestyle="-")

##

pgm.add_edge("Acur", "AAnext", linestyle="-")
pgm.add_edge("AAnext", "Anext", linestyle="-")
pgm.add_edge("Lcur", "LLnext", linestyle="-")
pgm.add_edge("LLnext", "Lnext", linestyle="-")

pgm.add_edge("XAcur", "AAnext", linestyle="-")
pgm.add_edge("QLcur", "AAnext", linestyle="-")

pgm.add_edge("AAnext", "QAnext", linestyle="-")
pgm.add_edge("QAnext", "ZAnext", linestyle="-")
pgm.add_edge("ZAnext", "XAnext", linestyle="-")
pgm.add_edge("XAnext", "XnextPred", linestyle="-")
#pgm.add_edge("XAnext", "LLnext", linestyle="-")
pgm.add_edge("LLnext", "QLnext", linestyle="-")
pgm.add_edge("QLnext", "ZLnext", linestyle="-")
pgm.add_edge("ZLnext", "XnextPred", linestyle="-")

pgm.add_edge("ZLnext", "Lnext", linestyle="-")
pgm.add_edge("ZAnext", "Anext", linestyle="-")


pgm.render()
fname = "vnem"
fcurer = "/Users/kpmurphy/github/pyprobml/figures"
pgm.figure.savefig(os.path.join(fcurer, "{}.png".format(fname)))
#pgm.figure.savefig(os.path.join(folder, "{}.pdf".format(fname)))


#input('done')
#exit()



###

import os
#import imp
import daft
#daft = imp.load_source('daft', 'daft-080308/daft.py')

objName1 = "Z1"
objStr1 = r"$Z_1$"
objName2 = "Z2"
objStr2 = r"$Z_2$"
objName3 = "Z3"
objStr3 = r"$Z_3$"

loc2dName1 = "Y1"
loc2dStr1 = r"$Y_1$"
loc2dName2 = "Y2"
loc2dStr2 = r"$Y_2$"

if 0:
    pgm = daft.PGM([4, 4], origin=[1, 0], observed_style="inner")
    pgm.add_node(daft.Node("Z1", r"$Z_1$", 2, 3))
    pgm.add_node(daft.Node("Z2", r"$Z_2$", 3, 3))
    pgm.add_node(daft.Node("Z3", r"$Z_3$", 4, 3))
    pgm.add_node(daft.Node("F", r"$F$", 3, 1))
    
    pgm.add_edge("F", "Z1")
    pgm.add_edge("F", "Z2")
    pgm.add_edge("F", "Z3")
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-classifier"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
if 1:
    pgm = daft.PGM([4, 4], origin=[1, 0], observed_style="inner")
    
    pgm.add_node(daft.Node("Z1", r"$Z_1$", 2, 3))
    pgm.add_node(daft.Node("Z2", r"$Z_2$", 3, 3))
    pgm.add_node(daft.Node("Z3", r"$Z_3$", 4, 3))
    pgm.add_node(daft.Node("F", r"$F$", 3, 1))
    pgm.add_node(daft.Node("Y1", r"$Y_1$", 2, 2))
    pgm.add_node(daft.Node("Y2", r"$Y_2$", 4, 2))
    
    
    pgm.add_edge("F", "Y1")
    pgm.add_edge("F", "Y2")
    
    pgm.add_edge("Y1", "Z1")
    pgm.add_edge("Y2", "Z1")
    pgm.add_edge("F", "Z1")
    
    pgm.add_edge("Y1", "Z2")
    pgm.add_edge("Y2", "Z2")
    pgm.add_edge("F", "Z2")
    
    pgm.add_edge("Y1", "Z3")
    pgm.add_edge("Y2", "Z3")
    pgm.add_edge("F", "Z3")
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-det"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
 
if 1:   
    pgm = daft.PGM([5,5], origin=[1, 0], observed_style="inner")
    
    pgm.add_node(daft.Node("Z1", r"$Z_1$", 2, 4))
    pgm.add_node(daft.Node("Z2", r"$Z_2$", 3, 4))
    pgm.add_node(daft.Node("Z3", r"$Z_3$", 4, 4))
    pgm.add_node(daft.Node("H", r"$H$", 3, 3))
    pgm.add_node(daft.Node("Hold", r"$H_{t-1}$", 1.5, 3))
    pgm.add_node(daft.Node("F", r"$F$", 3, 1))
    pgm.add_node(daft.Node("Y1", r"$Y_1$", 2, 2))
    pgm.add_node(daft.Node("Y2", r"$Y_2$", 4, 2))
    
    
    pgm.add_edge("F", "Y1")
    pgm.add_edge("F", "Y2")
    
    pgm.add_edge("Y1", "H")
    pgm.add_edge("Y2", "H")
    pgm.add_edge("F", "H")
    pgm.add_edge("Hold", "H")
    
    pgm.add_edge("H", "Z1")
    pgm.add_edge("H", "Z2")
    pgm.add_edge("H", "Z3")
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-det-rnn"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
    

if 1:
    pgm = daft.PGM([9, 6], origin=[0, 0], observed_style="inner")
    
    pgm.add_node(daft.Node("Z1", r"$Z_1$", 2, 5))
    pgm.add_node(daft.Node("Z2", r"$Z_2$", 3, 5))
    pgm.add_node(daft.Node("Z3", r"$Z_3$", 4, 5))
    pgm.add_node(daft.Node("F", r"$F$", 3, 1))
    pgm.add_node(daft.Node("Y1", r"$Y_1$", 2, 2))
    pgm.add_node(daft.Node("Y2", r"$Y_2$", 4, 2))
    pgm.add_node(daft.Node("C", r"$C$", 1, 3))
    pgm.add_node(daft.Node("X1", r"$X_1$", 2, 4))
    pgm.add_node(daft.Node("I1", r"$I_1$", 3, 4))
    pgm.add_node(daft.Node("X2", r"$X_2$", 4, 4))
    pgm.add_node(daft.Node("I2", r"$I_2$", 5, 4))
    #pgm.add_node(daft.Node("N", r"$N=2$", 1, 4))
    pgm.add_node(daft.Node("Z1old", r"$Z_1^{t-1}$", 0.5, 5))
    
    pgm.add_edge("F", "I1")
    pgm.add_edge("F", "I2")
    pgm.add_edge("Y1", "X1")
    pgm.add_edge("Y2", "X2")
    pgm.add_edge("C", "X1")
    pgm.add_edge("C", "X2")
    pgm.add_edge("F", "Y1")
    pgm.add_edge("F", "Y2")
    #pgm.add_edge("F", "N")
    pgm.add_edge("F", "C")
    
    pgm.add_edge("X1", "Z1")
    pgm.add_edge("X2", "Z1")
    pgm.add_edge("I1", "Z1")
    pgm.add_edge("I2", "Z1")
    pgm.add_edge("Z1old", "Z1")
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-da-2d"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
if 1:
    pgm = daft.PGM([9, 6], origin=[0, 0], observed_style="inner")
    
    pgm.add_node(daft.Node("Z1", r"$Z_1$", 2, 5))
    pgm.add_node(daft.Node("Z2", r"$Z_2$", 3, 5))
    pgm.add_node(daft.Node("Z3", r"$Z_3$", 4, 5))
    pgm.add_node(daft.Node("F", r"$F$", 3, 1))
    pgm.add_node(daft.Node("Y1", r"$Y_1$", 2, 2))
    pgm.add_node(daft.Node("Y2", r"$Y_2$", 4, 2))
    pgm.add_node(daft.Node("C", r"$C$", 1, 3))
    pgm.add_node(daft.Node("X1", r"$X_1$", 2, 4))
    pgm.add_node(daft.Node("I1", r"$I_1$", 3, 4))
    pgm.add_node(daft.Node("X2", r"$X_2$", 4, 4))
    pgm.add_node(daft.Node("I2", r"$I_2$", 5, 4))
    #pgm.add_node(daft.Node("N", r"$N=2$", 1, 4))
    pgm.add_node(daft.Node("Z1old", r"$Z_1^{t-1}$", 0.5, 5))
    
    pgm.add_edge("F", "I1")
    pgm.add_edge("F", "I2")
    pgm.add_edge("Y1", "X1")
    pgm.add_edge("Y2", "X2")
    pgm.add_edge("C", "X1")
    pgm.add_edge("C", "X2")
    pgm.add_edge("F", "Y1")
    pgm.add_edge("F", "Y2")
    #pgm.add_edge("F", "N")
    pgm.add_edge("F", "C")
    
    pgm.add_edge("X1", "Z1")
    pgm.add_edge("X2", "Z1")
    pgm.add_edge("I1", "Z1")
    pgm.add_edge("I2", "Z1")
    pgm.add_edge("Z1old", "Z1")
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-da-3d"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
 
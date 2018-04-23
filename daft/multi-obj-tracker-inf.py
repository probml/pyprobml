import os
#import imp
import daft
#daft = imp.load_source('daft', 'daft-080308/daft.py')

obj1 = "Z1"
obj1Str = r"$O_1$"
obj2 = "Z2"
obj2Str = r"$O_2$"
obj3 = "Z3"
obj3Str = r"$O_3$"
obj1Old = "Z1old"
obj1OldStr = r"$O_1^{t-1}$"



loc2d1 = "Y1"
loc2d1Str = r"$X_1$"
loc2d2 = "Y2"
loc2d2Str = r"$X_2$"

loc3d1 = "X1"
loc3d1Str = r"$Z_1$"
loc3d2 = "X2"
loc3d2Str = r"$Z_2$"

frame = "F"
frameStr = r"$F$"

#figs = ["classifier", "det", "det-rnn", "det-rnn-3d", "da-2d", "da-3d"]
figs = ["det-rnn-3d"]

if "classifier" in figs:
    pgm = daft.PGM([4, 4], origin=[1, 0], observed_style="inner")
    pgm.add_node(daft.Node(obj1, obj1Str, 2, 3))
    pgm.add_node(daft.Node(obj2, obj2Str, 3, 3))
    pgm.add_node(daft.Node(obj3, obj3Str, 4, 3))
    pgm.add_node(daft.Node(frame, frameStr, 3, 1))
    
    pgm.add_edge(frame, obj1)
    pgm.add_edge(frame, obj2)
    pgm.add_edge(frame, obj3)
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-classifier"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
if "det" in figs:
    pgm = daft.PGM([4, 4], origin=[1, 0], observed_style="inner")
    
    pgm.add_node(daft.Node(obj1, obj1Str, 2, 3))
    pgm.add_node(daft.Node(obj2, obj2Str, 3, 3))
    pgm.add_node(daft.Node(obj3, obj3Str, 4, 3))
    pgm.add_node(daft.Node(frame, frameStr, 3, 1))
    pgm.add_node(daft.Node(loc2d1, loc2d1Str, 2, 2))
    pgm.add_node(daft.Node(loc2d2, loc2d2Str, 4, 2))
    
    
    pgm.add_edge(frame, loc2d1)
    pgm.add_edge(frame, loc2d2)
    
    pgm.add_edge(loc2d1, obj1)
    pgm.add_edge(loc2d2, obj1)
    pgm.add_edge(frame, obj1)
    
    pgm.add_edge(loc2d1, obj2)
    pgm.add_edge(loc2d2, obj2)
    pgm.add_edge(frame, obj2)
    
    pgm.add_edge(loc2d1, obj3)
    pgm.add_edge(loc2d2, obj3)
    pgm.add_edge(frame, obj3)
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-det"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
 
if "det-rnn" in figs:   
    pgm = daft.PGM([5,5], origin=[1, 0], observed_style="inner")
    
    pgm.add_node(daft.Node(obj1, obj1Str, 2, 4))
    pgm.add_node(daft.Node(obj2, obj2Str, 3, 4))
    pgm.add_node(daft.Node(obj3, obj3Str, 4, 4))
    pgm.add_node(daft.Node("H", r"$H$", 3, 3))
    pgm.add_node(daft.Node("Hold", r"$H_{t-1}$", 1.5, 3))
    pgm.add_node(daft.Node(frame, frameStr, 3, 1))
    pgm.add_node(daft.Node(loc2d1, loc2d1Str, 2, 2))
    pgm.add_node(daft.Node(loc2d2, loc2d2Str, 4, 2))
    
    
    pgm.add_edge(frame, loc2d1)
    pgm.add_edge(frame, loc2d2)
    
    pgm.add_edge(loc2d1, "H")
    pgm.add_edge(loc2d2, "H")
    pgm.add_edge(frame, "H")
    pgm.add_edge("Hold", "H")
    
    pgm.add_edge("H", obj1)
    pgm.add_edge("H", obj2)
    pgm.add_edge("H", obj3)
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-det-rnn"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
    


    
if "da-2d" in figs:
    pgm = daft.PGM([9, 6], origin=[0, 0], observed_style="inner")
    
    pgm.add_node(daft.Node(obj1, obj1Str, 2, 5))
    pgm.add_node(daft.Node(obj2, obj2Str, 3, 5))
    pgm.add_node(daft.Node(obj3, obj3Str, 4, 5))
    pgm.add_node(daft.Node(frame, frameStr, 3, 1))
    pgm.add_node(daft.Node(loc2d1, loc2d1Str, 2, 2))
    pgm.add_node(daft.Node(loc2d2, loc2d2Str, 4, 2))
    #pgm.add_node(daft.Node("C", r"$C$", 1, 3))
    #pgm.add_node(daft.Node(loc3d1, loc3d1Str, 2, 4))
    pgm.add_node(daft.Node("I1", r"$I_1$", 3, 4))
    #pgm.add_node(daft.Node(loc3d2, loc3d2Str, 4, 4))
    pgm.add_node(daft.Node("I2", r"$I_2$", 5, 4))
    #pgm.add_node(daft.Node("N", r"$N=2$", 1, 4))
    pgm.add_node(daft.Node(obj1Old, obj1OldStr, 0.5, 5))
    
    pgm.add_edge(frame, "I1")
    pgm.add_edge(frame, "I2")
    #pgm.add_edge(loc2d1, loc3d1)
    #pgm.add_edge(loc2d2, loc3d2)
    #pgm.add_edge("C", loc3d1)
    #pgm.add_edge("C", loc3d2)
    pgm.add_edge(frame, loc2d1)
    pgm.add_edge(frame, loc2d2)
    #pgm.add_edge(frame, "N")
    #pgm.add_edge(frame, "C")
    
    pgm.add_edge(loc2d1, obj1)
    pgm.add_edge(loc2d2, obj1)
    pgm.add_edge("I1", obj1)
    pgm.add_edge("I2", obj1)
    pgm.add_edge(obj1Old, obj1)
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-da-2d"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
if "da-3d" in figs:
    pgm = daft.PGM([9, 6], origin=[0, 0], observed_style="inner")
    
    pgm.add_node(daft.Node(obj1, obj1Str, 2, 5))
    pgm.add_node(daft.Node(obj2, obj2Str, 3, 5))
    pgm.add_node(daft.Node(obj3, obj3Str, 4, 5))
    pgm.add_node(daft.Node(frame, frameStr, 3, 1))
    pgm.add_node(daft.Node(loc2d1, loc2d1Str, 2, 2))
    pgm.add_node(daft.Node(loc2d2, loc2d2Str, 4, 2))
    pgm.add_node(daft.Node("C", r"$C$", 1, 3))
    pgm.add_node(daft.Node(loc3d1, loc3d1Str, 2, 4))
    pgm.add_node(daft.Node("I1", r"$I_1$", 3, 4))
    pgm.add_node(daft.Node(loc3d2, loc3d2Str, 4, 4))
    pgm.add_node(daft.Node("I2", r"$I_2$", 5, 4))
    #pgm.add_node(daft.Node("N", r"$N=2$", 1, 4))
    pgm.add_node(daft.Node(obj1Old, obj1OldStr, 0.5, 5))
    
    pgm.add_edge(frame, "I1")
    pgm.add_edge(frame, "I2")
    pgm.add_edge(loc2d1, loc3d1)
    pgm.add_edge(loc2d2, loc3d2)
    pgm.add_edge("C", loc3d1)
    pgm.add_edge("C", loc3d2)
    pgm.add_edge(frame, loc2d1)
    pgm.add_edge(frame, loc2d2)
    #pgm.add_edge(frame, "N")
    pgm.add_edge(frame, "C")
    
    pgm.add_edge(loc3d1, obj1)
    pgm.add_edge(loc3d2, obj1)
    pgm.add_edge("I1", obj1)
    pgm.add_edge("I2", obj1)
    pgm.add_edge(obj1Old, obj1)
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-da-3d"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
if "det-rnn-3d" in figs:
    pgm = daft.PGM([9, 7], origin=[0, 0], observed_style="inner")
    
    pgm.add_node(daft.Node(obj1, obj1Str, 2, 6))
    pgm.add_node(daft.Node(obj2, obj2Str, 3, 6))
    pgm.add_node(daft.Node(obj3, obj3Str, 4, 6))
    pgm.add_node(daft.Node(frame, frameStr, 3, 1))
    pgm.add_node(daft.Node(loc2d1, loc2d1Str, 2, 2))
    pgm.add_node(daft.Node(loc2d2, loc2d2Str, 4, 2))
    pgm.add_node(daft.Node("C", r"$C$", 1, 3))
    pgm.add_node(daft.Node(loc3d1, loc3d1Str, 2, 4))
    #pgm.add_node(daft.Node("I1", r"$I_1$", 3, 4))
    pgm.add_node(daft.Node(loc3d2, loc3d2Str, 4, 4))
    #pgm.add_node(daft.Node("I2", r"$I_2$", 5, 4))
    #pgm.add_node(daft.Node("N", r"$N=2$", 1, 4))
    #pgm.add_node(daft.Node(obj1Old, obj1OldStr, 0.5, 5))
    
    pgm.add_node(daft.Node("H", r"$H$", 3, 5))
    pgm.add_node(daft.Node("Hold", r"$H_{t-1}$", 1.5, 5))
    
    #pgm.add_edge(frame, "I1")
    #pgm.add_edge(frame, "I2")
    pgm.add_edge(loc2d1, loc3d1)
    pgm.add_edge(loc2d2, loc3d2)
    pgm.add_edge("C", loc3d1)
    pgm.add_edge("C", loc3d2)
    pgm.add_edge(frame, loc2d1)
    pgm.add_edge(frame, loc2d2)
    #pgm.add_edge(frame, "N")
    pgm.add_edge(frame, "C")
    
    pgm.add_edge(loc3d1, "H")
    pgm.add_edge(loc3d2, "H")
    pgm.add_edge("H", obj1)
    pgm.add_edge("H", obj2)
    pgm.add_edge("H", obj3)
    pgm.add_edge("Hold", "H")
    
    pgm.render()
    folder = "/Users/kpmurphy/github/pyprobml/figures"
    fname = "multi-obj-tracker-det-rnn-3d"
    pgm.figure.savefig(os.path.join(folder, "{}.png".format(fname)))
    
 
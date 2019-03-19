#http://nbviewer.jupyter.org/urls/sf.net/p/pyx/gallery/connect/attachment/connect.ipynb

import os
import sys
root = "/Users/kpmurphy/github/pyprobml"
folder = os.path.join(root, "figures")


fname = os.path.join(root, 'pyx/tools/pyxtools.py')
#import imp
#imp.load_source('tools', fname)
exec(open(fname).read())

#sys.path.append(os.path.abspath("/Users/kpmurphy/github/pyprobml/pyx"))
#from tools.pyxtools import *
#import tools.pyxtools


unit.set(uscale=3)
c = canvas.canvas()


(c, Z) = add_text_circle(c, "Z", -0.5, 0, attr=[style.linestyle.dashed])
(c, A) = add_text_circle(c, "A", 0, 0, shaded=True, attr=[style.linestyle.dashed])
(c, B) = add_text_diamond(c, "B", 1, 0, attr=[style.linestyle.dashed, style.linewidth(0.01)])
(c, C) = add_text_box(c, "C", 1, 1, shaded=True)
(c, D) = add_text_box(c, "D", 0, 1)
(c, E) = add_text_diamond(c, " ", 0.5, 0.5, shaded=True,  attr=[style.linestyle.dashed])

c = connect(c, A, B, 0, txt="$x_2^3$", ydelta = 0.05, 
            attr=[color.rgb.red, style.linestyle.dotted, style.linewidth(0.01)])
c = connect(c, A, C, 45, txt = "foo", xdelta = 0.1,  attr=[style.linestyle.dashed])
c = connect(c, A, D, -45)

                
fname = 'pyx-demo'   
fname = os.path.join(folder, "{}.pdf".format(fname))      
c.writePDFfile(fname)



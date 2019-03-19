import os
import sys
#root = "/Users/kpmurphy/github/pyprobml"
root = os.getcwd()
figfolder = os.path.join(root, "figures")

fname = os.path.join(root, 'pyx/tools/pyxtools.py')
exec(open(fname).read())

unit.set(uscale=3)

c = canvas.canvas()

(c, h1) = add_text_diamond(c, "$h_{t-1}$", 1, 2, xdelta=-0.1)
(c, h2) = add_text_diamond(c, "$h_{t}$", 2, 2)
(c, x) = add_text_circle(c, "$x_{t}$", 2, 1, xdelta=-0.05)
(c, z) = add_text_circle(c, "$z_{t}$", 2, 3, xdelta=-0.05)

c = connect(c, h1, z)
                
fname = 'VRNN-prior'   
fname = os.path.join(figfolder, "{}.pdf".format(fname))      
c.writePDFfile(fname)


c = canvas.canvas()

(c, h1) = add_text_diamond(c, "$h_{t-1}$", 1, 2, xdelta=-0.1)
(c, h2) = add_text_diamond(c, "$h_{t}$", 2, 2)
(c, x) = add_text_circle(c, "$x_{t}$", 2, 1, xdelta=-0.05)
(c, z) = add_text_circle(c, "$z_{t}$", 2, 3, xdelta=-0.05)

c = connect(c, h1, x)
c = connect(c, z, x, angle=45)
                
fname = 'VRNN-gen'   
fname = os.path.join(figfolder, "{}.pdf".format(fname))      
c.writePDFfile(fname)


c = canvas.canvas()

(c, h1) = add_text_diamond(c, "$h_{t-1}$", 1, 2, xdelta=-0.1)
(c, h2) = add_text_diamond(c, "$h_{t}$", 2, 2)
(c, x) = add_text_circle(c, "$x_{t}$", 2, 1, xdelta=-0.05)
(c, z) = add_text_circle(c, "$z_{t}$", 2, 3, xdelta=-0.05)

c = connect(c, h1, h2)
c = connect(c, x, h2)
c = connect(c, z, h2)
                
fname = 'VRNN-recur'   
fname = os.path.join(figfolder, "{}.pdf".format(fname))      
c.writePDFfile(fname)


c = canvas.canvas()

(c, h1) = add_text_diamond(c, "$h_{t-1}$", 1, 2, xdelta=0)
(c, h2) = add_text_diamond(c, "$h_{t}$", 2, 2, xdelta=0)
(c, x) = add_text_circle(c, "$x_{t}$", 2, 1, xdelta=-0.05)
(c, z) = add_text_circle(c, "$z_{t}$", 2, 3, xdelta=-0.05)

c = connect(c, h1, h2)
c = connect(c, h1, z)
c = connect(c, h1, x)
c = connect(c, x, h2)
c = connect(c, z, h2)
c = connect(c, z, x, angle=45)

                
fname = 'VRNN-full'   
fname = os.path.join(figfolder, "{}.pdf".format(fname))      
c.writePDFfile(fname)


c = canvas.canvas()

(c, h1) = add_text_diamond(c, "$h_{t-1}$", 1, 2, xdelta=-0.1)
(c, h2) = add_text_diamond(c, "$h_{t}$", 2, 2)
(c, x) = add_text_circle(c, "$x_{t}$", 2, 1, xdelta=-0.05)
(c, z) = add_text_circle(c, "$z_{t}$", 2, 3, xdelta=-0.05)

c = connect(c, h1, z, attr=[style.linestyle.dotted])
c = connect(c, x, z, angle=45, attr=[style.linestyle.dotted])

                
fname = 'VRNN-inf'   
fname = os.path.join(figfolder, "{}.pdf".format(fname))      
c.writePDFfile(fname)



c = canvas.canvas()

(c, h1) = add_text_diamond(c, "$h_{t-1}$", 1, 2, xdelta=-0.1)
(c, h2) = add_text_diamond(c, "$h_{t}$", 2, 2, xdelta=-0.1)
(c, x) = add_text_circle(c, "$x_{t}$", 2, 1, xdelta=-0.1)

c = connect(c, h1, h2)
c = connect(c, h1, x)
c = connect(c, x, h2)
                
fname = 'VRNN-RNN'   
fname = os.path.join(figfolder, "{}.pdf".format(fname))      
c.writePDFfile(fname)




c = canvas.canvas()

(c, h1) = add_text_circle(c, "$z_{t-1}$", 1, 2, xdelta=-0.1)
(c, x1) = add_text_circle(c, "$x_{t-1}$", 1, 1, xdelta=-0.1)
(c, h2) = add_text_circle(c, "$z_{t}$", 2, 2, xdelta=-0.05)
(c, x2) = add_text_circle(c, "$x_{t}$", 2, 1, xdelta=-0.05)

c = connect(c, h1, h2)
c = connect(c, h1, x1)
c = connect(c, h2, x2)
                
fname = 'VRNN-SSM'   
fname = os.path.join(figfolder, "{}.pdf".format(fname))      
c.writePDFfile(fname)


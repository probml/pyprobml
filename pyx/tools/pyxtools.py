# http://pyx.sourceforge.net/

from math import sin, cos, pi
from pyx import *
from pyx.connector import arc, curve, line

def solid_dot(x, y):
   return path.circle(x, y, 0.1)
   
    
def make_polygon(r, n):
    return box.polygon([(-r*sin(i*2*pi/n), r*cos(i*2*pi/n))
                            for i in range(n)])
         
def add_text_box(c, str, x, y, shaded=False, attr=[]):
    bb = text.text(x, y, str)
    p = bb.bbox().enlarged(0.1).path()
    if shaded:
        a = [deco.filled([color.grey(0.85)])]
        c.fill(p, a)
    c.stroke(p, attr)
    c.insert(bb)
    return (c, bb)
    
    
def add_text_box1(c, str, x, y, shaded=False, attr=[]):
    bb = text.text(x, y, str)
    if shaded:
        attr = attr + [deco.filled([color.grey(0.85)])]
    c.draw(bb.bbox().enlarged(0.1).path(), attr + [deco.stroked()])
    c.insert(bb)
    return (c, bb)
    
def add_text_diamond(c, str, x, y, shaded=False, attr=[]):
    bb = text.text(x-0.05, y, str)
    bb2 = make_polygon(0.2, 4)
    p = bb2.path()
    p = p.transformed(trafo.translate(x, y))
    if shaded:
        a = [deco.filled([color.grey(0.85)])]
        c.fill(p, a)
    c.stroke(p, attr)
    c.insert(bb)
    return (c, bb)

    
def add_text_circle(c, str, x, y, r=0.15, shaded=False, attr=[]):
    bb = text.text(x-0.05, y, str)
    p = path.circle(x, y, r)
    c.stroke(p, attr)
    if shaded:
        a = [deco.filled([color.grey(0.85)])]
        c.fill(p, a)
    c.insert(bb)
    return (c, bb)
    

def connect(c, X, Y, angle=0, txt=[], attr=[], xdelta=0, ydelta=0):
    if angle == 0:
        p = line(X, Y, boxdists=[0.2, 0.2])
    else:
        p = arc(X, Y,  boxdists=[0.2, 0.2], relangle = angle)
    c.stroke(p, attr + [deco.earrow.normal])
    if txt:
        loc = p.at(0.5*p.arclen())
        #c.fill(solid_dot(*p.at(0.5*p.arclen())))
        bb = text.text(loc[0] + xdelta, loc[1] + ydelta, txt)
        c.insert(bb)
    return c
        
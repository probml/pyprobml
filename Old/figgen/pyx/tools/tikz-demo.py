import tikz

import os
import sys
#root = "/Users/kpmurphy/github/pyprobml"
root = os.getcwd()
figfolder = os.path.join(root, "figures")

fname = os.path.join(root, 'pyx/tools/tikz.py')
exec(open(fname).read())

def demo1():
    pic = Picture()

    # define some internal TikZ coordinates
    pic['pointA'] = 5 + 0j      # complex number for x-y coordinates
    pic['pointB'] = '45:5'      # or string for any native TikZ format

    # define a coordinate in Python
    pointC = 3 + 1j

    # draw new path
    with pic.path('draw') as draw:
        draw.at(0 + 0j, name='start').grid_to(6 + 6j)

    with pic.path('fill, color=red') as draw:
        draw.at('start')\
            .line_to('pointA', coord='relative')\
            .line_to('pointB').node('hello', style='above, black')\
            .line_to(pointC)\
            .spline_to(5 + 1j, 2 + 5j).node('spline')\
            .cycle()            # close path

    with pic.path('fill, color=blue') as draw:
        draw.at(pointC).circle(0.3)

    print(pic.make())

def demo2():
    # http://www.texample.net/tikz/examples/glider/
    pic = tikz.Picture('thick')
    
    circle_size = 0.42
    circle_pos = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 2)]
    
    with pic.path('draw') as draw:
        draw.at(0 + 0j).grid_to(3 + 3j)
    
    with pic.path('fill') as fill:
        for pos in circle_pos:
            fill.at(pos).circle(circle_size)
    
    print(pic.make())


def demo3():
    pic = tikz.Picture()
    
    circle_size = 0.2
    pic['A'] = 0 + 0j
    pic['B'] = 0 + 1j
    
    with pic.path('draw') as draw:
            draw.at('A').circle(circle_size)
            draw.at('B').circle(circle_size)
            draw.at('A').line_to('B')
            draw.at(1 + 1j).node(name='C', text='CC')
            draw.at('C').circle(circle_size)
            draw.at('A').line_to('C')
    
    print(pic.make())



pic = tikz.Picture()

circle_size = 0.2
N = 3
A = 0; B = 1; C = 2;
locs = [0 + 0j, 0 + 1j, 1 + 1j]
names = ['A', 'B', 'CC']

with pic.path('draw') as draw:
    for idx in range(N):
        loc = locs[idx]
        name = names[idx]
        draw.at(loc).node(name = name, text = name)
        draw.at(name).circle(circle_size)
    draw.at(names[A]).line_to(names[B])
    draw.at(names[A]).line_to(names[C], type='hv')
    draw.at(names[B]).arc_to(names[C])

print(pic.make())

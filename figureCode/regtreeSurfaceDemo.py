import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import os

t1 = 5
t3 = 3
t2 = 7
t4 = 3

r = np.linspace(2, 10, 5)

#A function to return a tree for given (x1,x2) coordinates
def ManualTree(x1, x2):
    if x1 <= t1:
        if x2 <= t2:
            z = r[0]
        else:
            if x1 <= t3:
                z = r[3]
            else:
                z = r[4]
    else:
        if x2 <= t4:
            z = r[1]
        else:
            z = r[2]
    return(z)

ManualTree = np.vectorize(ManualTree)

x = np.linspace(0,10,100)
X, Y = np.meshgrid(x, x)
Z = ManualTree(X.T,Y.T)

#A 3D matrix for determining which colors go where.

def DivList(list1, den):
    return([e/den for e in list1])

#This tells us which color we use for which output tree value. Intended to match with the latex tree graphic.
def ColorMapper(z):
    if z == r[0]:
        out = DivList([255.0, 0.0, 0.0], 255.0)
    elif z == r[1]:
        out = DivList([0.0, 0.0, 255.0], 255.0)
    elif z == r[2]:
        out = DivList([160.0, 32.0, 240.0], 255.0)
    elif z == r[3]:
        out = DivList([0.0, 100.0, 0.0], 255.0)
    else:
        out = DivList([255.0, 140.0, 0.0], 255.0)
    return(out)

#Manually build the tree, one output tree value at a top. 
#For some spots, we need to add in the walls to show difference between two tree values.
fig = plt.figure(figsize=(20.0/1.8, 15.0/1.8))
ax = fig.add_subplot(111, projection='3d')
for val in r:   
    if val in [2, 4, 8]:
        logi = Z == val
        if val == 2:
            logi[50,0:70] = True
            logi[:51,70] = True
            shp = (51, 71)
        elif val == 4:
            logi[50:,30] = True
            shp = (50, 31)
        else:
            logi[30,70:] = True
            shp = (31, 30)
        x = X[logi].reshape(shp)
        y = Y[logi].reshape(shp)
        z = Z[logi].reshape(shp)
    else:
        x = X[Z==val]
        y = Y[Z==val]
        z = val
    ax.plot_wireframe(x, y, z, color=ColorMapper(val))
        
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.view_init(elev=30, azim=230)
plt.savefig(os.path.join('figures', 'tree3d.pdf'))

# https://commons.wikimedia.org/wiki/File:Rosenbrock_function.svg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = Axes3D(fig, azim=-128, elev=43)
s = .05
X = np.arange(-2, 2.+s, s)
Y = np.arange(-1, 3.+s, s)
X, Y = np.meshgrid(X, Y)
#Z = (1.-X)**2 + 100.*(Y-X*X)**2
Z = (1.-X)**2 + 5.*(Y-X*X)**2
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, norm = LogNorm(),
#                 cmap="viridis")
# Without using `` linewidth=0, edgecolor='none' '', the code may produce a
# graph with wide black edges, which will make the surface look much darker
# than the one illustrated in the figure above.
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, norm=LogNorm(),
                linewidth=0, edgecolor='none', cmap="viridis")

# Set the axis limits so that they are the same as in the figure above.
ax.set_xlim([-2, 2.0])                                                       
ax.set_ylim([-1, 3.0])                                                       
#ax.set_zlim([0, 2500]) 

plt.xlabel("x")
plt.ylabel("y")
#plt.savefig("figures/rosen.pdf", bbox_inches="tight")
plt.savefig("figures/rosen-a1b5.pdf", bbox_inches="tight")

plt.show()
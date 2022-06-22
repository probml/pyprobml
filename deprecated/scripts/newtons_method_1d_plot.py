import superimport

import matplotlib.pyplot as plt
import numpy as np

import pyprobml_utils as pml

xmin = 0.1
xmax = 12
ymin= -5
ymax = 4
domain = np.linspace(xmin, xmax, 1191) # num=1191 assumes a step size of 0.01 for this domain
f = lambda x: np.log(x) - 2

x_k = 2
m = 1/x_k
b = f(x_k) - m*x_k
tl = lambda x: m*x + b

plt.plot((0.1, 12), (0,0), '-k', linewidth=2, zorder=1)
plt.plot(domain, f(domain), '-r', linewidth=3, label=r"$g(x)$", zorder=2)
plt.plot(domain, tl(domain), '--b', linewidth=2.5, label=r"$g_{lin}(x)$", zorder=3)

plt.scatter(x_k, f(x_k), marker='.', c='black', s=180, zorder=4)
plt.scatter(-b/m, 0, marker='.', c='black', s=180, zorder=4)
plt.plot((x_k, x_k), (ymin, f(x_k)), ":k")
plt.plot((-b/m, -b/m), (ymin, 0), ":k")

plt.xlim((xmin, xmax))
plt.ylim((ymin, ymax))

frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])

plt.xticks([x_k, -b/m], [r'$x_{k}$', r'$x_{k} + d_{k}$'])


plt.legend()
pml.savefig("newtonsMethodMin1d.pdf")
plt.show()


import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

xs = np.arange(-2, 2, step=0.01)
y1 = xs ** 2
y2 = y1 - 1
y3 = np.abs(y2)
y4 = y3 + y1
plt.figure()
plt.plot(xs, y1, 'r-', label='x^2')
plt.plot(xs, y2, 'g:', label='x^2 - 1')
plt.plot(xs, y3, 'b-', label='|x^2 - 1|')
plt.plot(xs, y4, 'k-', label='|x^2 - 1| + x^2')
plt.legend()
save_fig('weaklyConvex.png')
plt.show()

def f(x):
    return np.abs(x**2 - 1)

def grad(xp):
    return 2*xp

def lin(x, xp):
    g = grad(xp)
    return np.abs(f(xp) - (x-xp)*g)


ys = [f(x) for x in xs]
x1 = 0.5
x2 = -1.0
zs1 = [lin(x, x1) for x in xs]
zs2 = [lin(x, x2) for x in xs]
plt.figure()
plt.plot(xs, ys, 'b', label='f')
plt.plot(xs, zs1, 'r--', label='$f_{x_1}$')
plt.plot(xs, zs2, 'r-', label='$f_{x_2}$')
plt.axhline(y=0)
plt.ylim([-0.5, 3])
plt.annotate('$(x_1, f(x_1))$', xy=(0.5, 0.75))
plt.annotate('$(x_2, f(x_2))$', xy=(-1, -0.2))
plt.legend(loc='upper right', fontsize=14)
save_fig('proxLinear.png')
plt.show()

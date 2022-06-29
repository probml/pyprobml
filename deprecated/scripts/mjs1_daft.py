import superimport

import daft

import matplotlib.patches as patches

# Colors.
p_color = {"ec": "#46a546"}
s_color = {"ec": "#f89406"}
r_color = {"ec": "#dc143c"}

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

pgm = daft.PGM(shape=(5, 5), origin=(-1,-1))

N=3
for i in range(N):
    print("x{}".format(i))
    pgm.add_node("x{}".format(i), r"$x_{}$".format(i), i, 1, observed=True)
    #if i>0:
    #    pgm.add_edge("x0", "x{}".format(i))

pgm.add_edge("x0", "x1")
params = {'connectionstyle': patches.ConnectionStyle.Arc3(rad=0.2)}
pgm.add_edge("x0", "x2", plot_params=params)

pgm.render()
pgm.savefig('../figures/mjs1_pgm.pdf')
pgm.show()
import superimport

import daft

# Colors.
p_color = {"ec": "#46a546"}
s_color = {"ec": "#f89406"}
r_color = {"ec": "#dc143c"}

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

pgm = daft.PGM(shape=(12, 5), origin=(-1,0))

pgm.add_node("Ax", r"$\theta^h$", 2.5, 3, observed=False)
for i in range(4):
    pgm.add_node("x{}".format(i), r"$x_{}$".format(i), i+1, 1, observed=True)
    pgm.add_node("hx{}".format(i), r"$h^x_{}$".format(i), i + 1, 2, observed=False)
    pgm.add_edge("Ax", "hx{}".format(i))
    pgm.add_edge("hx{}".format(i), "x{}".format(i))
    if i>0:
        pgm.add_edge("hx{}".format(i - 1), "hx{}".format(i))


pgm.add_node("Ay", r"$\theta^h$", 7.5, 3, observed=False)
delta = 5
for i in range(4):
    pgm.add_node("y{}".format(i), r"$y_{}$".format(i), i+1+delta, 1, observed=True)
    pgm.add_node("hy{}".format(i), r"$h^y_{}$".format(i), i + 1+delta, 2, observed=False)
    pgm.add_edge("Ay", "hy{}".format(i))
    pgm.add_edge("hy{}".format(i), "y{}".format(i))
    if i>0:
        pgm.add_edge("hy{}".format(i - 1), "hy{}".format(i))


pgm.add_node("z", r"$z$", 5, 4, observed=False)
pgm.add_edge("z", "Ax")
pgm.add_edge("z", "Ay")

pgm.add_node("thetax", r"$\theta^x$", 0, 1, observed=False)
pgm.add_node("thetay", r"$\theta^y$", 10, 1, observed=False)
pgm.add_edge("thetax", "x0")
pgm.add_edge("thetay", "y3")


pgm.render()
pgm.savefig('../figures/visual_spelling_hmm_pgm.pdf')
pgm.show()
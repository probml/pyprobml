"""
n-body particle inference
=========================

Dude.
"""

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

pgm = daft.PGM([5.4, 2.0], origin=[0.65, 0.35])

kx, ky = 1.5, 1.
nx, ny = kx + 3., ky + 0.
hx, hy, dhx = kx - 0.5, ky + 1., 1.

pgm.add_node(daft.Node("dyn", r"$\theta_{\mathrm{dyn}}$", hx + 0. * dhx, hy + 0.))
pgm.add_node(daft.Node("ic", r"$\theta_{\mathrm{I.C.}}$", hx + 1. * dhx, hy + 0.))
pgm.add_node(daft.Node("sun", r"$\theta_{\odot}$",        hx + 2. * dhx, hy + 0.))
pgm.add_node(daft.Node("bg", r"$\theta_{\mathrm{bg}}$",   hx + 3. * dhx, hy + 0.))
pgm.add_node(daft.Node("Sigma", r"$\Sigma^2$",            hx + 4. * dhx, hy + 0.))

pgm.add_plate(daft.Plate([kx - 0.5, ky - 0.6, 2., 1.1], label=r"model points $k$"))
pgm.add_node(daft.Node("xk", r"$x_k$", kx + 0., ky + 0.))
pgm.add_edge("dyn", "xk")
pgm.add_edge("ic", "xk")
pgm.add_node(daft.Node("yk", r"$y_k$", kx + 1., ky + 0.))
pgm.add_edge("sun", "yk")
pgm.add_edge("xk", "yk")

pgm.add_plate(daft.Plate([nx - 0.5, ny - 0.6, 2., 1.1], label=r"data points $n$"))
pgm.add_node(daft.Node("sigman", r"$\sigma^2_n$", nx + 1., ny + 0., observed=True))
pgm.add_node(daft.Node("Yn", r"$Y_n$", nx + 0., ny + 0., observed=True))
pgm.add_edge("bg", "Yn")
pgm.add_edge("Sigma", "Yn")
pgm.add_edge("Sigma", "Yn")
pgm.add_edge("yk", "Yn")
pgm.add_edge("sigman", "Yn")

# Render and save.
pgm.render()
pgm.figure.savefig("huey_p_newton.pdf")
pgm.figure.savefig("huey_p_newton.png", dpi=150)

"""
The GALEX Photon Catalog
========================

This is the Hogg \& Schiminovich model for how photons turn into
counts in the GALEX satellite data stream.  Note the use of relative
positioning.

"""

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft
pgm = daft.PGM([5.4, 5.4], origin=[1.2, 1.2])
wide = 1.5
verywide = 1.5 * wide
dy = 0.75

# electrons
el_x, el_y = 2., 2.
pgm.add_plate(daft.Plate([el_x - 0.6, el_y - 0.6, 2.2, 2 * dy + 0.3], label="electrons $i$"))
pgm.add_node(daft.Node("xabc", r"xa$_i$,xabc$_i$,ya$_i$,\textit{etc}", el_x + 0.5, el_y + 0 * dy, aspect=2.3 * wide, observed=True))
pgm.add_node(daft.Node("xyti", r"$x_i,y_i,t_i$", el_x + 1., el_y + 1 * dy, aspect=wide))
pgm.add_edge("xyti", "xabc")

# intensity fields
ph_x, ph_y = el_x + 2.5, el_y + 3 * dy
pgm.add_node(daft.Node("Ixyt", r"$I_{\nu}(x,y,t)$", ph_x, ph_y, aspect=verywide))
pgm.add_edge("Ixyt", "xyti")
pgm.add_node(daft.Node("Ixnt", r"$I_{\nu}(\xi,\eta,t)$", ph_x, ph_y + 1 * dy, aspect=verywide))
pgm.add_edge("Ixnt", "Ixyt")
pgm.add_node(daft.Node("Iadt", r"$I_{\nu}(\alpha,\delta,t)$", ph_x, ph_y + 2 * dy, aspect=verywide))
pgm.add_edge("Iadt", "Ixnt")

# s/c
sc_x, sc_y = ph_x + 1.5, ph_y - 1.5 * dy
pgm.add_node(daft.Node("dark", r"dark", sc_x, sc_y - 1 * dy, aspect=wide))
pgm.add_edge("dark", "xyti")
pgm.add_node(daft.Node("flat", r"flat", sc_x, sc_y, aspect=wide))
pgm.add_edge("flat", "xyti")
pgm.add_node(daft.Node("att", r"att", sc_x, sc_y + 3 * dy))
pgm.add_edge("att", "Ixnt")
pgm.add_node(daft.Node("optics", r"optics", sc_x, sc_y + 2 * dy, aspect=wide))
pgm.add_edge("optics", "Ixyt")
pgm.add_node(daft.Node("psf", r"psf", sc_x, sc_y + 1 * dy))
pgm.add_edge("psf", "xyti")
pgm.add_node(daft.Node("fee", r"f.e.e.", sc_x, sc_y - 2 * dy, aspect=wide))
pgm.add_edge("fee", "xabc")

# sky
pgm.add_node(daft.Node("sky", r"sky", sc_x, sc_y + 4 * dy))
pgm.add_edge("sky", "Iadt")

# stars
star_x, star_y = el_x, el_y + 4 * dy
pgm.add_plate(daft.Plate([star_x - 0.6, star_y - 0.6, 2.2, 2 * dy + 0.3], label="stars $n$"))
pgm.add_node(daft.Node("star adt", r"$I_{\nu,n}(\alpha,\delta,t)$", star_x + 0.5, star_y + 1 * dy, aspect=verywide))
pgm.add_edge("star adt", "Iadt")
pgm.add_node(daft.Node("star L", r"$L_{\nu,n}(t)$", star_x + 1, star_y, aspect=wide))
pgm.add_edge("star L", "star adt")
pgm.add_node(daft.Node("star pos", r"$\vec{x_n}$", star_x, star_y))
pgm.add_edge("star pos", "star adt")

# done
pgm.render()
pgm.figure.savefig("galex.pdf")
pgm.figure.savefig("galex.png", dpi=150)

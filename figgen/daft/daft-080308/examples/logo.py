#!/usr/bin/env python
"""
That's an awfully DAFT logo!

"""

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import daft


if __name__ == "__main__":
    # Instantiate the PGM.
    pgm = daft.PGM((3.7, 0.7), origin=(0.15, 0.15))

    pgm.add_node(daft.Node("d", r"$D$", 0.5, 0.5))
    pgm.add_node(daft.Node("a", r"$a$", 1.5, 0.5, observed=True))
    pgm.add_node(daft.Node("f", r"$f$", 2.5, 0.5))
    pgm.add_node(daft.Node("t", r"$t$", 3.5, 0.5))

    pgm.add_edge("d", "a")
    pgm.add_edge("a", "f")
    pgm.add_edge("f", "t")

    pgm.render()
    pgm.figure.savefig("logo.pdf")
    pgm.figure.savefig("logo.png", dpi=200, transparent=True)

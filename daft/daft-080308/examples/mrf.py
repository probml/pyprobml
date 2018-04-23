"""
An undirected graph
===================

This makes the simple point that you don't have to have directions on
your edges; you can have *undirected* graphs.  (Also, the nodes don't
need to have labels!)

"""

import itertools
import numpy as np

import daft

# Instantiate the PGM.
pgm = daft.PGM([3.6, 3.6], origin=[0.7, 0.7], node_unit=0.4, grid_unit=1,
        directed=False)

for i, (xi, yi) in enumerate(itertools.product(range(1, 5), range(1, 5))):
    pgm.add_node(daft.Node(str(i), "", xi, yi))


for e in [(4, 9), (6, 7), (3, 7), (10, 11), (10, 9), (10, 14),
        (10, 6), (10, 7), (1, 2), (1, 5), (1, 0), (1, 6), (8, 12), (12, 13),
        (13, 14), (15, 11)]:
    pgm.add_edge(str(e[0]), str(e[1]))

# Render and save.
pgm.render()
pgm.figure.savefig("mrf.pdf")
pgm.figure.savefig("mrf.png", dpi=150)

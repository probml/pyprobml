"""
Nodes can contain words
=======================

We here at **Daft** headquarters tend to put symbols (variable
names) in our graph nodes.  But you don't have to if you don't
want to.

"""

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

pgm = daft.PGM([3.6, 2.7], origin=[1.15, 0.65])
pgm.add_node(daft.Node("cloudy", r"cloudy", 3, 3, aspect=1.8))
pgm.add_node(daft.Node("rain", r"rain", 2, 2, aspect=1.2))
pgm.add_node(daft.Node("sprinkler", r"sprinkler", 4, 2, aspect=2.1))
pgm.add_node(daft.Node("wet", r"grass wet", 3, 1, aspect=2.4, observed=True))
pgm.add_edge("cloudy", "rain")
pgm.add_edge("cloudy", "sprinkler")
pgm.add_edge("rain", "wet")
pgm.add_edge("sprinkler", "wet")
pgm.render()
pgm.figure.savefig("wordy.pdf")
pgm.figure.savefig("wordy.png", dpi=150)

.. _nocircles:


Nodes can go free
=================

.. figure:: /_static/examples/nocircles.png


Nodes can go free
=================

You don't need to put ellipses or circles around your node contents,
if you don't want to.



::

    
    from matplotlib import rc
    rc("font", family="serif", size=12)
    rc("text", usetex=True)
    
    import daft
    
    pgm = daft.PGM([3.6, 2.4], origin = [1.15, 0.8], node_ec="none")
    pgm.add_node(daft.Node("cloudy", r"cloudy", 3, 3))
    pgm.add_node(daft.Node("rain", r"rain", 2, 2))
    pgm.add_node(daft.Node("sprinkler", r"sprinkler", 4, 2))
    pgm.add_node(daft.Node("wet", r"grass wet", 3, 1))
    pgm.add_edge("cloudy", "rain")
    pgm.add_edge("cloudy", "sprinkler")
    pgm.add_edge("rain", "wet")
    pgm.add_edge("sprinkler", "wet")
    pgm.render()
    pgm.figure.savefig("nocircles.pdf")
    pgm.figure.savefig("nocircles.png", dpi=150)
    


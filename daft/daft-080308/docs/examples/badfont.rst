.. _badfont:


You can use arbitrarily shitty fonts
====================================

.. figure:: /_static/examples/badfont.png


You can use arbitrarily shitty fonts
====================================

Any fonts that LaTeX or matplotlib supports can be used. Do not take
this example as any kind of implied recommendation unless you plan on
announcing a *huge* discovery!



::

    
    from matplotlib import rc
    
    ff = "comic sans ms"
    # ff = "impact"
    # ff = "times new roman"
    
    rc("font", family=ff, size=12)
    rc("text", usetex=False)
    
    import daft
    
    pgm = daft.PGM([3.6, 1.8], origin=[2.2, 1.6], aspect=2.1)
    pgm.add_node(daft.Node("confused", r"confused", 3.0, 3.0))
    pgm.add_node(daft.Node("ugly", r"ugly font", 3.0, 2.0, observed=True))
    pgm.add_node(daft.Node("bad", r"bad talk", 5.0, 2.0, observed=True))
    pgm.add_edge("confused", "ugly")
    pgm.add_edge("ugly", "bad")
    pgm.add_edge("confused", "bad")
    pgm.render()
    pgm.figure.savefig("badfont.pdf")
    pgm.figure.savefig("badfont.png", dpi=150)
    


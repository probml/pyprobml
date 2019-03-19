.. _weaklensing:


A model for weak lensing
========================

.. figure:: /_static/examples/weaklensing.png


A model for weak lensing
========================

This is (**Daft** co-author) Hogg's model for the obsevational
cosmology method known as *weak gravitational lensing*, if that method
were properly probabilistic (which it usually isn't).  Hogg put the
model here for one very important reason: *Because he can*.  Oh, and
it demonstrates that you can represent non-trivial scientific projects
with **Daft**.



::

    
    from matplotlib import rc
    rc("font", family="serif", size=12)
    rc("text", usetex=True)
    rc("./weaklensing.tex")
    
    import daft
    
    pgm = daft.PGM([4.7, 2.35], origin=[-1.35, 2.2])
    pgm.add_node(daft.Node("Omega", r"$\Omega$", -1, 4))
    pgm.add_node(daft.Node("gamma", r"$\gamma$", 0, 4))
    pgm.add_node(daft.Node("obs", r"$\epsilon^{\mathrm{obs}}_n$", 1, 4, observed=True))
    pgm.add_node(daft.Node("alpha", r"$\alpha$", 3, 4))
    pgm.add_node(daft.Node("true", r"$\epsilon^{\mathrm{true}}_n$", 2, 4))
    pgm.add_node(daft.Node("sigma", r"$\sigma_n$", 1, 3))
    pgm.add_node(daft.Node("Sigma", r"$\Sigma$", 0, 3))
    pgm.add_node(daft.Node("x", r"$x_n$", 2, 3, observed=True))
    pgm.add_plate(daft.Plate([0.5, 2.25, 2, 2.25],
            label=r"galaxies $n$"))
    pgm.add_edge("Omega", "gamma")
    pgm.add_edge("gamma", "obs")
    pgm.add_edge("alpha", "true")
    pgm.add_edge("true", "obs")
    pgm.add_edge("x", "obs")
    pgm.add_edge("Sigma", "sigma")
    pgm.add_edge("sigma", "obs")
    pgm.render()
    pgm.figure.savefig("weaklensing.pdf")
    pgm.figure.savefig("weaklensing.png", dpi=150)
    


import daft

pgm = daft.PGM([2, 1], observed_style="outer", aspect=3.2)
pgm.add_node(daft.Node("fixed", r"Fixed!", 1, 0.5, observed=True))
pgm.render().figure.savefig("fixed.png", dpi=150)

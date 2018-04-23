from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft

if __name__ == "__main__":
    pgm = daft.PGM([1.1, 3.15], origin=[0.45, 2.2])
    pgm.add_node(daft.Node("a", r"$a$", 1, 5))
    pgm.add_node(daft.Node("b", r"$b$", 1, 4))
    pgm.add_node(daft.Node("c", r"$c_n$", 1, 3, observed=True))
    pgm.add_plate(daft.Plate([0.5, 2.25, 1, 1.25], label=r"data $n$"))
    pgm.add_edge("a", "b")
    pgm.add_edge("b", "c")
    pgm.render()
    pgm.figure.savefig("bca.pdf")
    pgm.figure.savefig("bca.png", dpi=150)

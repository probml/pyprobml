from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft

if __name__ == "__main__":
    pgm = daft.PGM([3.7, 3.15], origin=[-0.35, 2.2])
    pgm.add_node(daft.Node("omega", r"$\omega$", 2, 5))
    pgm.add_node(daft.Node("true", r"$\tilde{X}_n$", 2, 4))
    pgm.add_node(daft.Node("obs", r"$X_n$", 2, 3, observed=True))
    pgm.add_node(daft.Node("alpha", r"$\alpha$", 3, 4))
    pgm.add_node(daft.Node("Sigma", r"$\Sigma$", 0, 3))
    pgm.add_node(daft.Node("sigma", r"$\sigma_n$", 1, 3))
    pgm.add_plate(daft.Plate([0.5, 2.25, 2, 2.25], label=r"stars $n$"))
    pgm.add_edge("omega", "true")
    pgm.add_edge("true", "obs")
    pgm.add_edge("alpha", "true")
    pgm.add_edge("Sigma", "sigma")
    pgm.add_edge("sigma", "obs")
    pgm.render()
    pgm.figure.savefig("gaia.pdf")
    pgm.figure.savefig("gaia.png", dpi=150)

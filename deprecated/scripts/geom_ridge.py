# Geometry of Ridge Regression

# Author: Gerardo Durán Martín

import superimport

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Filter warning creating parts of the ellipses that
# do not exist
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def range_chebyshev(a, b, steps):
    """
    Create a grid point of N+1 values
    """
    theta_vals = np.arange(steps+1) * np.pi / steps
    x_vals = (a + b) / 2 + (a - b) / 2 * np.cos(theta_vals)

    return x_vals


def plot_ellipses(x0, y0, a, b, r, lim=5, steps=10000, **kwargs):
    """
    Plot an ellipses of the form
        a(x - x0) ^ 2 + b(y - y0) ^ 2 = r^2
    """
    xrange = range_chebyshev(-lim, lim, steps)
    yrange_up = np.sqrt(b * (r ** 2 - a * (xrange - x0) ** 2)) / b + y0
    yrange_down = -np.sqrt(b * (r ** 2 - a * (xrange - x0) ** 2)) / b + y0
    plt.plot(xrange, yrange_up, **kwargs)
    plt.plot(xrange, yrange_down, **kwargs)


def main():
    plot_ellipses(0, 0, 1, 4, 4, c="tab:red")
    plot_ellipses(-3, -2, 1, 1, 2, c="tab:green")
    plt.scatter(0, 0, marker="x", s=50, c="tab:red")
    plt.scatter(-3, -2, marker="x", s=50, c="tab:green")
    plt.scatter(-1.7, -0.2, c="tab:purple")
    plt.text(0.1, 0.1, "ML Estimate")
    plt.text(-2.7, -2.4, "prior mean")
    plt.text(-1.6, -0.3, "MAP Estimate",
             verticalalignment="top", horizontalalignment="left")

    plt.text(3, 0.3, r"$u_1$", verticalalignment="bottom", fontsize=13)
    plt.text(0.3, 3, r"$u_2$", horizontalalignment="left", fontsize=13)
    plt.hlines(0, 0, 3)
    plt.vlines(0, 0, 3)
    plt.xlim(-5, 5)
    plt.axis("equal")
    plt.axis("off")
    plt.savefig("../figures/geom_ridge.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

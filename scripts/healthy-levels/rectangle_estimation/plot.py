import numpy as np
import matplotlib.pyplot as plt


class MatplotlibMixin:

    def plot(self):
        self.ax = plt.gca()
        self._plot()
        plt.title(self.title, fontsize=12, y=1.03)
        plt.savefig(self.filename)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    def plot_rectangle(self, rectangle, color):
        l1, l2, s1, s2 = rectangle
        self.ax.add_patch(
                plt.Rectangle((l1, l2),
                              s1, s2, fill=False,
                              edgecolor=color, linewidth=3)
                )

    def plot_points(self, points):
        xs = points[:, 0]
        ys = points[:, 1]
        self.ax.scatter(xs, ys, marker='+', color='red', zorder=10, linewidth=3)

    def plot_contours(self, xy, z):
        x = xy[:, 0]
        y = xy[:, 1]
        dim = int(np.sqrt(len(x)))
        xx = x.reshape(dim, dim)
        yy = y.reshape(dim, dim)
        zz = z.reshape(dim, dim)

        #x = np.arange(0.01, 1+0.01, 0.01)
        #xs, ys = np.meshgrid(x, x)
        #zz = z.reshape(xx.shape)
        self.ax.contour(xx, yy, zz)

    def colors_from_proba(self, probabilities):
        """ Returns greyscale colours that reflect the supplied relative probabilities,
        ranging from black for the most probable, to light grey for the least probable."""
        max_prob = np.max(probabilities)
        intensities = 1 - (0.25 + 0.75 * probabilities / max_prob)
        intensities = intensities.reshape(intensities.shape + (1,))
        
        # Repeat the same intensity for all RGB channels.
        return np.repeat(intensities, 3, intensities.ndim - 1)

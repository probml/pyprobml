
#import superimport

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D

from inspect import getsourcefile
from os.path import abspath


#https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python?lq=1
def get_current_path():
    current_path = abspath(getsourcefile(lambda:0)) # fullname of current file
    #current_path = os.path.dirname(__file__)
    current_dir = os.path.dirname(current_path)
    return current_dir

def test():
    print('welcome to python probabilistic ML library')
    print(get_current_path())

# https://stackoverflow.com/questions/10685495/reducing-the-size-of-pdf-figure-file-in-matplotlib

def save_fig(fname, *args, **kwargs):
    #figdir = '../figures' # default directory one above where code lives
    current_dir = get_current_path()
    figdir = os.path.join(current_dir, "..", "figures")

    if not os.path.exists(figdir):
        print('making directory {}'.format(figdir))
        os.mkdir(figdir)

    fname_full = os.path.join(figdir, fname)
    print('saving image to {}'.format(fname_full))
    #plt.tight_layout()

    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams['pdf.fonttype'] = 42

    # Font sizes
    SIZE_SMALL = 12
    SIZE_MEDIUM = 14
    SIZE_LARGE = 24
    # https://stackoverflow.com/a/39566040
    plt.rc('font', size=SIZE_SMALL)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE_SMALL)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE_SMALL)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE_SMALL)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE_SMALL)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE_SMALL)  # legend fontsize
    plt.rc('figure', titlesize=SIZE_LARGE)  # fontsize of the figure title

    plt.savefig(fname_full, *args, **kwargs)
    
    
def savefig(fname, *args, **kwargs):
    save_fig(fname, *args, **kwargs)
    
from matplotlib.patches import Ellipse, transforms
# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def plot_ellipse(Sigma, mu, ax, n_std=3.0, facecolor='none', edgecolor='k', plot_center='true', **kwargs):
    cov = Sigma
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = (transforms.Affine2D()
                        .rotate_deg(45)
                        .scale(scale_x, scale_y)
                        .translate(mean_x, mean_y))

    ellipse.set_transform(transf + ax.transData)

    if plot_center:
        ax.plot(mean_x, mean_y, '.')
    return ax.add_patch(ellipse)

def plot_ellipse_test():
    fig, ax = plt.subplots()
    Sigma = np.array([[5,1],[1,5]])
    plot_ellipse(Sigma, np.zeros(2), ax, n_std=1)
    plt.axis('equal')
    plt.show()


def convergence_test(fval, previous_fval, threshold=1e-4, warn=False):
    eps = 2e-10
    converged = 0
    delta_fval = np.abs(fval - previous_fval)
    avg_fval = (np.abs(fval) + abs(previous_fval) + eps) / 2.0
    if (delta_fval / avg_fval) < threshold:
        converged = 1

    if warn and (fval - previous_fval) < -2 * eps:
        print('convergenceTest:fvalDecrease', 'objective decreased!')
    return converged

def hinton_diagram(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('white')
    ax.set_aspect('equal', 'box')

    for (x, y), w in np.ndenumerate(matrix):
        color = 'lawngreen' if w > 0 else 'royalblue'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    nr, nc = matrix.shape
    ax.set_xticks(np.arange(0, nr))
    ax.set_yticks(np.arange(0, nc))
    ax.grid(linestyle='--', linewidth=2)
    ax.autoscale_view()
    ax.invert_yaxis()


def kdeg(x, X, h):
    """
    KDE under a gaussian kernel

    Parameters
    ----------
    x: array(eval, D)
    X: array(obs, D)
    h: float

    Returns
    -------
    array(eval):
        KDE around the observed values
    """
    N, D = X.shape
    nden, _ = x.shape

    Xhat = X.reshape(D, 1, N)
    xhat = x.reshape(D, nden, 1)
    u = xhat - Xhat
    u = linalg.norm(u, ord=2, axis=0) ** 2 / (2 * h ** 2)
    px = np.exp(-u).sum(axis=1) / (N * h * np.sqrt(2 * np.pi))
    return px


def scale_3d(ax, x_scale, y_scale, z_scale, factor):
    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=factor
    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)
    return short_proj


def style3d(ax, x_scale, y_scale, z_scale, factor=0.62):
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.get_proj = scale_3d(ax, x_scale, y_scale, z_scale, factor)


if __name__ == "__main__":
    test()
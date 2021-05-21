
import os
import matplotlib.pyplot as plt
import numpy as np

def test():
    print('welcome to python probabilistic ML library')


# https://stackoverflow.com/questions/10685495/reducing-the-size-of-pdf-figure-file-in-matplotlib
    
def save_fig(fname, *args, **kwargs):
    '''Save current plot window to the figures directory.'''
    if "PYPROBML" in os.environ:
        root = os.environ["PYPROBML"]
        figdir = os.path.join(root, 'figures')
    else:
        figdir = '../figures' # default directory one above where code lives
    if not os.path.exists(figdir):
        os.mkdir(figdir)
    fname_full = os.path.join(figdir, fname)
    print('saving image to {}'.format(fname_full))
    #plt.tight_layout()
    plt.savefig(fname_full,  dpi=300)
    
    
def savefig(fname, *args, **kwargs):
    save_fig(fname, *args, **kwargs)

from matplotlib.patches import Ellipse, transforms
# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def plot_ellipse(Sigma, mu, ax, n_std=3.0, facecolor='none', edgecolor='k',  **kwargs):
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

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

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

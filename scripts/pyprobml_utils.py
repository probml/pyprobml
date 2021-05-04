
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def test():
    print('welcome to python probabilistic ML library')

def compute_image_resize(image, width = None, height = None):
     # From https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
     # initialize the dimensions of the image to be resized and
     # grab the image size
     dim = None
     (h, w) = image.shape[:2]

     # check to see if the width is None
     if width is None:
         # calculate the ratio of the height and construct the
         # dimensions
         r = height / float(h)
         dim = (int(w * r), height)

     # otherwise, the height is None
     else:
         # calculate the ratio of the width and construct the
         # dimensions
         r = width / float(w)
         dim = (width, int(h * r))

     return dim
    
    
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
    plt.tight_layout()
    plt.savefig(fname_full, *args, **kwargs)
    
    
def savefig(fname, *args, **kwargs):
    save_fig(fname, *args, **kwargs)


def git_ssh(git_command, email, username, verbose=False):
    '''Execute a git command via ssh from colab.
    Details in https://github.com/probml/pyprobml/blob/master/book1/intro/colab_intro.ipynb
    Authors: Mahmoud Soliman <mjs@aucegypt.edu> and Kevin Murphy <murphyk@gmail.com>
    '''
    git_command=git_command.replace(r"https://github.com/","git@github.com:")
    print('executing command via ssh:', git_command)
    # copy keys from drive to local .ssh folder
    if verbose:
        print('Copying keys from gdrive to local VM')
    os.system('rm -rf ~/.ssh')
    os.system('mkdir ~/.ssh')
    os.system('cp  -r /content/drive/MyDrive/ssh/* ~/.ssh/')
    os.system('ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts')
    os.system('ssh -T git@github.com') # test
    # git commands
    if verbose:
        print('Executing git commands')
    os.system('git config --global user.email {}'.format(email))
    os.system('git config --global user.name {}'.format(username))
    os.system(git_command)
    # cleanup
    if verbose:
        print('Cleanup local VM')
    os.system('rm -r ~/.ssh/')
    os.system('git config --global user.email ""')
    os.system('git config --global user.name ""')
    

# Source:
# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


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

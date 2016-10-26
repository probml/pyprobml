#!/usr/bin/env python3
# Miscellaneous utility functions

import os
import scipy.io as sio
import numpy as np
import glob
from init_pmtk3 import DATA_DIR
#from demos import linreg_1d_batch_demo 
#import demos.linreg_1d_batch_demo
#from demos.linreg_1d_batch_demo import main

def nsubplots(n):
    '''Returns [ynum, xnum], which  how many plots in the y and x directions to
    cover n in total while keeping the aspect ratio close to rectangular'''
    if n==2:
        ynum = 2; xnum = 2;
    else:
        xnum = np.ceil(np.sqrt(n));
        ynum = np.ceil(n/np.float(xnum));
    return ynum, xnum

def add_ones(X):
    """Add a column of ones to X"""
    n = len(X)
    return np.column_stack((np.ones(n), X))


def degexpand(X, deg, add_ones=False):
    """Expand input vectors to contain powers of the input features"""
    n = len(X)
    xx = X
    for i in xrange(1, deg):
        xx = np.column_stack((xx, np.power(X, i + 1)))
    if add_ones:
        xx = np.column_stack((np.ones(n), xx))

    return xx


def rescale_data(X, min_val=-1, max_val=1, minx=None, rangex=None):
    """
    Rescale columns to lie in the range
    [min_val, max_val] (defaults to [-1,1]])
    """
    if minx is None:
        minx = X.min(axis=0)
    if rangex is None:
        rangex = X.max(axis=0) - X.min(axis=0)

    return (max_val - min_val) * (X - minx) / rangex + min_val


def center_cols(X, mu=None):
    """
    Make each column be zero mean
    """
    if mu is None:
        mu = X.mean(axis=0)
    return X - mu, mu


def mk_unit_variance(X, s=None):
    """
    Make each column of X be variance 1
    """
    if s is None:
        s = X.std(axis=0)

    try:
        len(s)
        s[s < np.spacing(1)] = 1
    except TypeError:
        s = s if s > np.spacing(1) else 1

    return X / s, s


class preprocessor_create():
    def __init__(self, standardize_X=False, rescale_X=False, kernel_fn=None,
                 poly=None, add_ones=False):
        self.standardize_X = standardize_X
        self.rescale_X = rescale_X
        self.kernel_fn = kernel_fn
        self.poly = poly
        self.add_ones = add_ones


def poly_data_make(sampling="sparse", deg=3, n=21):
    """
    Create an artificial dataset
    """
    np.random.seed(0)

    if sampling == "irregular":
        xtrain = np.concatenate(
            (np.arange(-1, -0.5, 0.1), np.arange(3, 3.5, 0.1)))
    elif sampling == "sparse":
        xtrain = np.array([-3, -2, 0, 2, 3])
    elif sampling == "dense":
        xtrain = np.arange(-5, 5, 0.6)
    elif sampling == "thibaux":
        xtrain = np.linspace(0, 20, n)
        xtest = np.arange(0, 20, 0.1)
        sigma2 = 4
        w = np.array([-1.5, 1/9.])
        fun = lambda x: w[0]*x + w[1]*np.square(x)

    if sampling != "thibaux":
        assert deg < 4, "bad degree, dude %d" % deg
        xtest = np.arange(-7, 7, 0.1)
        if deg == 2:
            fun = lambda x: (10 + x + np.square(x))
        else:
            fun = lambda x: (10 + x + np.power(x, 3))
        sigma2 = np.square(5)

    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
        np.sqrt(sigma2)
    ytestNoisefree = fun(xtest)
    ytestNoisy = ytestNoisefree + np.random.normal(0, 1, xtest.shape) * \
        np.sqrt(sigma2)

    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2        


def load_mat(matName):
    """look for the .mat file in pmtk3/pmtkdataCopy/
    currently only support .mat files create by Matlab 5,6,7~7.2,
    """
    print 'looking for ', matName, ' in ', DATA_DIR
    try:
        data = sio.loadmat(os.path.join(DATA_DIR, matName))
    except NotImplementedError:
        raise
    except FileNotFoundError:
        raise
    return data


def generate_rst():
    """generate chX.rst in current working directory"""
    cwd = os.getcwd()
    demo_dir = os.path.join(cwd, 'demos')
    chapters = os.listdir(demo_dir)
    for chapter in chapters:
        if not os.path.isdir(os.path.join(demo_dir, chapter)):
            continue
        reg_py = os.path.join(demo_dir, chapter, '*.py')
        scripts = glob.glob(reg_py)
        rst_file = chapter + '.rst'
        rst_file = os.path.join(demo_dir, chapter, rst_file)
        with open(rst_file, 'w') as f:
            f.write(chapter)
            f.write('\n========================================\n')
            for script in scripts:
                script_name = os.path.basename(script)
                f.write('\n' + script_name[:-3])
                f.write('\n----------------------------------------\n')
                reg_png = os.path.join(demo_dir,
                                       chapter,
                                       script_name[:-3] + '*.png')
                for img in glob.glob(reg_png):
                    img_name = os.path.basename(img)
                    f.write(".. image:: " + img_name + "\n")
                f.write(".. literalinclude:: " + script_name + "\n")

#if __name__ == '__main__':
    #generate_rst()
    #print("Finished generate chX.rst!")
    #demos.linreg_1d_batch_demo.main()
    #linreg_1d_batch_demo.main()
    #main()

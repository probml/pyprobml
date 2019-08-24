
import os
import matplotlib.pyplot as plt

def pyprobml_test():
    print('welcome to python probabilistic ML library')

figdir = "../figures"
def save_fig(fname):
    try:
        print('saving figure {}'.format(fname))
        plt.savefig(os.path.join(figdir, fname))
    except:
        print('did not save figure {}, since figdir not defined'.format(fname))
        
        
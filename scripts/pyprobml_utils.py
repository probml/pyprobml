
import os
import matplotlib as plt

figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))
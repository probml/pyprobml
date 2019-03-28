import matplotlib.pyplot as plt
import os

def save_fig(fname):
    try:
        path = os.environ["PYPROBML"]
        figdir = os.path.join(path, "figures")
        plt.tight_layout()    
        fullname = os.path.join(figdir, fname)
        print('saving to {}'.format(fullname))
        plt.savefig(fullname)
    except:
        print("Not saving {}".format(fname))
    
import matplotlib.pyplot as plt
import os

def save_fig(fname):
    try:
        path = os.environ["PYPROBML"]
        figdir = os.path.join(path, "figures")
        plt.tight_layout()    
        fullname = os.path.join(figdir, fname)
        print('saving to {}'.format(fullname))
        # bbox_inches="tight" used to help at least numbers_game.py plot titles to be displayed correctly in PDFs.
        plt.savefig(fullname, bbox_inches="tight")
    except:
        print("Not saving {}".format(fname))
    
    
def get_data_dir():
    try:
        path = os.environ["PYPROBML"]
        data_dir = os.path.join(path, "data")
        return data_dir
    except:
        print("Cannot find data directory: environment variable PYPROBML not defined")
        raise
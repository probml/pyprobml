
import os
import matplotlib.pyplot as plt


def test():
    print('welcome to python probabilistic ML library')


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
    




def git_ssh(git_command, email="murphyk@gmail.com", username="probml",
            verbose=False):
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

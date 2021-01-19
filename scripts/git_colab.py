# Execute a git command via ssh from colab
# Details in https://github.com/probml/pyprobml/blob/master/book1/intro/colab_intro.ipynb
# Authors: Mahmoud Soliman <mjs@aucegypt.edu> and Kevin Murphy <murphyk@gmail.com>

import os

def git_ssh(git_command, email="murphyk@gmail.com", username="probml",
            verbose=False):
    git_command=git_command.replace(r"https://github.com/","git@github.com:")
    print('executing command:', git_command)
    # copy keys from drive to local .ssh folder
    os.system('ls -l')
    
   
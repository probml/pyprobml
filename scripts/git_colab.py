# Execute a git command via ssh from colab
# Details in https://github.com/probml/pyprobml/blob/master/book1/intro/colab_intro.ipynb
# Authors: Mahmoud Soliman <mjs@aucegypt.edu> and Kevin Murphy <murphyk@gmail.com>

import os

def git_ssh(git_command, email="murphyk@gmail.com", username="probml",
            verbose=False):
    git_command=git_command.replace(r"https://github.com/","git@github.com:")
    print('executing command via ssh:', git_command)
    # copy keys from drive to local .ssh folder
    print(verbose)
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

def main():
    git_ssh('this-will-fail')

if __name__ == "__main__":
    main()
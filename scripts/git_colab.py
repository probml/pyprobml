# Execute a git command via ssh from colab
# Details in https://github.com/probml/pyprobml/blob/master/book1/intro/colab_intro.ipynb
# Authors: Mahmoud Soliman <mjs@aucegypt.edu> and Kevin Murphy <murphyk@gmail.com>

import os

def git_ssh(git_command, email="murphyk@gmail.com", username="probml",
            verbose=False):
    git_command=git_command.replace(r"https://github.com/","git@github.com:")
    print('executing command:', git_command)
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
    os.system('git config --global user.email {}'.format(email)
    os.system('git config --global user.name {}'.format(username)
    os.system(git_command)
    # cleanup
    if verbose:
        print('Cleanup local VM')
    os.system('rm -r ~/.ssh/')
    os.system('git config --global user.email ""')
    os.system('git config --global user.name ""')
    
'''   
# This version uses colab magics.
def git_colab(
    git_command, email="murphyk@gmail.com", username="probml", verbose=False):
  git_command=git_command.replace(r"https://github.com/","git@github.com:")
  print('executing command:', git_command)
  # copy keys from drive to local .ssh folder
  if verbose:
    print('Copying keys from gdrive to local VM')
  !rm -rf ~/.ssh/
  !mkdir ~/.ssh/
  !cp  -r /content/drive/MyDrive/ssh/* ~/.ssh/
  if verbose:
    !ls ~/.ssh/
  # configure ssh and test it
  if verbose:
    print('Setup SSH')
  !ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
  !ssh -T git@github.com
  # git commands
  if verbose:
    print('Executing git commands')
  !git config --global user.email $email
  !git config --global user.name $username
  !$git_command
  # cleanup
  if verbose:
    print('Cleanup local VM')
  !rm -r ~/.ssh/
  !git config --global user.email ""
  !git config --global user.name ""
  # check that cleanup worked
  #!ssh -T git@github.com # should say 'Host key verification failed'
'''
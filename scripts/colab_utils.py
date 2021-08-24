import superimport

import os
import cv2 # open CV 2

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
 
    
def image_resize(img_path,size=None,ratio=None):
    if not size:
        size=[0,480]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    dim = compute_image_resize(img, height=size[1])
    img = cv2.resize(img, dim, interpolation =  cv2.INTER_AREA)
    return img


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
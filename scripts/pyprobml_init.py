import os
# edit this variable to point to where you download pyprobml
#os.environ["PYPROBML"] = "/Users/kpmurphy/github/pyprobml"
#figdir = os.path.join(os.environ["PYPROBML"], "figures")

#exec(open('utils.py').read())

#https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python?lq=1
from inspect import getsourcefile
from os.path import abspath
current_path = abspath(getsourcefile(lambda:0)) # fullname of current file
current_dir = os.path.dirname(current_path)
figdir = os.path.join(current_dir, "..", "figures")
print(figdir)
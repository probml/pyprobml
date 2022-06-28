import os
from re import L
import shutil

for chap_no in range(35,7,-1):
    # rename chap_no folders
    path = "notebooks/book2/"
    #print(os.path.join(path))
    shutil.move(path + f"{chap_no:02d}" + "/", path + f"{chap_no+1:02d}" + "/")


#

fid = open('matcode.txt', 'r')
lines = fid.readlines()
fid.close()

import re
txt = lines[-1]

def parse(txt):
  chapter_full = re.search("^([-\w]*)\.tex", txt)[0]
  chapter = chapter_full[:-4] # drop .tex
  filename_full = re.search("matcode{\w*}", txt)[0]
  filename = filename_full[8:-1]
  return chapter, filename

from collections import defaultdict

D = defaultdict(set)
for txt in lines:
  chapter, filename = parse(txt)
  D[filename].add(chapter)
  
fid = open('matcode.csv', 'w')
for key, val in D.items():
  matname = key
  for chapname in val:
    str = "{}.m \t {}.tex\n".format(matname, chapname)
    fid.write(str)
fid.close()
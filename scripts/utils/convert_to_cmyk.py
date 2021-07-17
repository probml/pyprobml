import pandas as pd
import glob
import re
import os

infile = '/Users/kpmurphy/github/bookv2/cmyk_convert.txt'
infolder = '/Users/kpmurphy/github/bookv2/figures'
outfolder = '/Users/kpmurphy/github/bookv2/figuresJPG'

df = pd.read_csv(infile, header=None, names=['Name'])
for i in range(len(df)):
    entry = df.loc[i]
    fname = entry['Name']

    # foo -> foo.pdf, but foo.png -> foo.png
    parts = fname.split('.')
    if len(parts)==1:
        fname = f'{fname}.pdf'

    print(fname)

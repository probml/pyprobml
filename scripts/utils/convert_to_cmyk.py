import pandas as pd
import glob
import re
import os

infile = '/Users/kpmurphy/github/bookv2/Misc/cmyk_convert.txt'
infolder = '/Users/kpmurphy/github/bookv2/figures'
outfolder = '/Users/kpmurphy/github/bookv2/figuresMagickPDF'

df = pd.read_csv(infile, header=None, names=['Name'])
for i in range(len(df)):
    entry = df.loc[i]
    fname = entry['Name'] # eg foo.png, or foo (assumed to be foo.pdf)
    parts = fname.split('.')
    stem = parts[0]
    if len(parts)==1: # no period, assumed to be pdf
        suffix = 'pdf'
    else:
        suffix = parts[1]
    src_name = f'{stem}.{suffix}'
    #dest_name = f'{stem}.jpg'
    dest_name = f'{stem}.pdf'
    source = f'{infolder}/{src_name}'
    dest = f'{outfolder}/{dest_name}'
    cmd = f'convert {source} -colorspace cmyk {dest}'
    print(cmd)
    os.system(cmd)
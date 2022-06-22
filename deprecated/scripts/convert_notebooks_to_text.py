import pandas as pd
import glob
import re
import os

root = '/Users/kpmurphy/github/probml-notebooks'
notebooks_dir = f'{root}/notebooks'
scripts_dir = f'{root}/notebooks-text-format'

filenames = glob.glob(f"{notebooks_dir}/*.ipynb")
for f in filenames:
    parts = f.split('/')
    name = parts[-1]
    name = name.replace('ipynb', 'py')
    fnew = f'{scripts_dir}/{name}'
    if os.path.exists(fnew):
        print('skipping ', f, 'since output already exists')
    else:
        cmd = f'jupytext --to py --output {fnew} {f}'
        print(cmd)
        os.system(cmd)

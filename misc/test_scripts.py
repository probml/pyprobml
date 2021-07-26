import os
import logging
import traceback
import subprocess

#logging.basicConfig(level=logging.DEBUG, filename='errors.log')
logf = open("errors.log", "w")

root = '/Users/kpmurphy/github/pyprobml/scripts'
filenames = ['linreg_poly_vs_degree.py', 'linreg_contours_sse_plot.py', 'linregOnlineDemo.py']

for f in filenames:
    python = 'python3'
    #python = '/opt/anaconda3/envs/spyder-dev/bin/python'
    cmd = f'{python} {root}/{f}'

    print('\n\n', cmd)
    try:
        os.system(cmd)
        #subprocess.run([cmd], check=True)
    except Exception as e:
        err = f'Failed to run {f}: {e}'
        print('******\n', err)
        logf.write(err)
        traceback.print_exc(file=logf)
        #logging.exception(err)
        pass


import superimport

import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
print('GPU info')
res = result.stdout;
print(res)


from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('!Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

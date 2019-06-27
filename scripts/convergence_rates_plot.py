# Plot theoretical rates of convergence

import numpy as np
import matplotlib.pyplot as plt
import os

def save_fig(fname):
    figdir = os.path.join(os.environ["PYPROBML"], "figures")
    plt.tight_layout()    
    fullname = os.path.join(figdir, fname)
    print('saving to {}'.format(fullname))
    plt.savefig(fullname)
    

plt.figure(figsize=(12,4))

ks = range(1,10)
ys = [1.0/k for k in ks]
print(ys)
plt.subplot(1,3,1)
plt.plot(ks, np.log(ys), color = 'r')
plt.title('Sublinear convergence')

ys = [1.0/(2**k) for k in ks]
print(ys)
plt.subplot(1,3,2)
plt.plot(ks, np.log(ys), color = 'g')
plt.title('Linear convergence')

ys = [1.0/(2**(2**k)) for k in ks]
print(ys)
plt.subplot(1,3,3)
plt.plot(ks, np.log(ys), color = 'b')
plt.title('Quadratic convergence')

#fig.subplots_adjust(hspace=0)
plt.tight_layout()
plt.draw()

fname = 'convergenceRates.pdf'
print(fname)
save_fig(fname)
plt.show()

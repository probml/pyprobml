#!/usr/bin/env python

# Plot the softmax function.

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import logsumexp
 
def softmax(a):
  e = np.exp((1.0 * np.array(a)))
  return e / np.sum(e)
  

T = [100, 2, 1]
a = np.array([3, 0, 1]);
ind = [1, 2, 3]

plt.figure(figsize=(12,4))
for i in range(len(T)):
    plt.subplot(1,3,i+1)
    plt.bar(ind, softmax(a / T[i]))
    plt.title('T = %d' % T[i])
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_xticklabels([])
    
plt.show()
fname = 'softmax_temp.pdf'
print(fname)
plt.savefig(os.path.join('../figures', fname))



#for i in range(len(T)):
#    plt.bar(ind, softmax(a / T[i]))
#    plt.ylim(0, 1)
#    plt.title('T = %d' % T[i])
#    plt.show()
#    #plt.savefig('softmax_temp%d.png' % T[i])
#    fname = 'softmax_temp%d.pdf' % T[i]
#    print(fname)
#    plt.savefig(os.path.join('../figures', fname))


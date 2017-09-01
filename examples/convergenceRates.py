import os
import matplotlib.pyplot as plt
import numpy as np

#http://matplotlib.org/examples/pylab_examples/subplots_demo.html

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
plt.show()
    
plt.show()
fname = 'convergenceRates.pdf'
print(fname)
plt.savefig(os.path.join('../figures', fname))


if 0:
    plt.figure(figsize=(12,4))
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    
    ks = range(1,10)
    ys = [1.0/k for k in ks]
    print(ys)
    ax1.plot(ks, np.log(ys), color = 'r')
    ax1.set_title('Sublinear convergence')
    
    ys = [1.0/(2**k) for k in ks]
    print(ys)
    ax2.plot(ks, np.log(ys), color = 'g')
    ax2.set_title('Linear convergence')
    
    ys = [1.0/(2**(2**k)) for k in ks]
    print(ys)
    ax3.plot(ks, np.log(ys), color = 'b')
    ax3.set_title('Quadratic convergence')
    
    #fig.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.draw()
    
    plt.savefig(os.path.join('../figures', 'convergenceRates.png'))
    
    plt.show()
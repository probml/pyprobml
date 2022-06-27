# Convert twoPowerCurves(Fig 5.15 (b)) to python/JAX
# Author: Garvit9000c

import superimport

import numpy as np
import matplotlib.pyplot as plt

pi=np.pi

#Domain
x=np.arange(-1*pi,pi,0.01)

#functions
y1=-1*np.cos(x)+0.03
y2=-1*np.cos(2*x/3)

plt.ylim(-2,2.5)

#Ploting Curve
plt.plot(x,y1,'b') #Curve_B
plt.plot(x,y2,'r') #Curve_A

#Small line indicating position of alpha
plt.hlines(y=-1.025, xmin=-0.3, xmax=0.3, linewidth=1, color='k')

#Axis Arrows
plt.arrow(0,-2.5,0,4.5,color='black',width = 0.03)       #y-axis 
plt.arrow(-1*pi,-1.5,2*pi,0,color='black',width = 0.03)  #x-axis

#labels
plt.text(-0.3, 2.2, r'$1 - \beta$', fontsize=15) # 1 - β
plt.text(-0.3, 0.5, '1', fontsize=15)     # 1
plt.text(-0.3, -1.3, r'$\alpha$', fontsize=15)    # α
plt.text(-0.34, -1.8, '$θ_0$', fontsize=12)  # θo
plt.text(3.1, -1.8, 'θ', fontsize=15)     # θ
plt.text(2.5, -0.2, 'A', fontsize=15)     # A
plt.text(2.5, 1.1, 'B', fontsize=15)      # B

plt.savefig('../figures/twoPowerCurves.pdf', dpi=300)
plt.show()

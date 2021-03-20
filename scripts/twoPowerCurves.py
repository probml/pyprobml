# Convert twoPowerCurves(Fig 5.15 (b)) to python/JAX
# Author: Garvit9000c

import jax.numpy as jnp
import matplotlib.pyplot as plt

pi=jnp.pi

#Domain
x=jnp.arange(-1*pi,pi,0.01)

#functions
y1=-1*jnp.cos(x)+0.03
y2=-1*jnp.cos(2*x/3)

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
plt.text(-0.3, 2.2, '1 - β', fontsize=15) # 1 - β
plt.text(-0.3, 0.5, '1', fontsize=15)     # 1
plt.text(-0.3, -1.3, 'α', fontsize=15)    # α
plt.text(-0.34, -1.8, 'θo', fontsize=12)  # θo
plt.text(3.1, -1.8, 'θ', fontsize=15)     # θ
plt.text(2.5, -0.2, 'A', fontsize=15)     # A
plt.text(2.5, 1.1, 'B', fontsize=15)      # B

plt.show()

import pyprobml_utils as pml
pml.save_fig('twoPowerCurves.pdf')

# Convert Neyman-Pearson testing paradigm(Fig 5.15 (a)) to python/JAX
# Author: Garvit9000c

import superimport

from scipy.stats import multivariate_normal as gaussprob
import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

#constants
pi=np.pi
sigma=1.5
xmin = -4
xmax = 8
ymin = 0
ymax = 0.3
res = 0.01

#Domain
x=np.arange(xmin,xmax,res)

#functions
y1=gaussprob.pdf(x, 0, sigma**2)
y2=gaussprob.pdf(x, 4, sigma**2)

#Axes Limits
plt.ylim(ymin,ymax)
plt.xlim(xmin,xmax)

#Ploting Curve
plt.plot(x,y1,'b') #Curve_B
plt.plot(x,y2,'r') #Curve_A


plt.vlines(x=2.3, ymin=0, ymax=0.5, linewidth=1.5, color='k')
plt.xticks([2.3],['$X^*$'],size=18)
plt.yticks([])

#Shading α Region
x1=np.arange(2.3,xmax,res)
y_1=gaussprob.pdf(x1, 0, sigma**2)
plt.fill_between(x1,y_1, 0, alpha=0.50)

#Shading β Region
x2=np.arange(xmin,2.3,res)
y_2=gaussprob.pdf(x2, 4, sigma**2)
plt.fill_between(x2,y_2, 0, alpha=0.50)

#Axis Arrows
plt.arrow(0,0.07,1.2,-0.05,color='black',head_width=0.02,head_length=0.2) #β      
plt.arrow(4,0.07,-1.2,-0.05,color='black',head_width=0.02,head_length=0.2)#α

#label
plt.text(-0.4, 0.07, r'$\beta$', fontsize=15)      #β
plt.text(4, 0.07, r'$\alpha$', fontsize=15)        #α
plt.text(-0.2, 0.28, '$H_0$', fontsize=15)         #H0
plt.text(3.8,0.28, '$H_1$', fontsize=15)           #H1

pml.savefig('neymanPearson2.pdf')
plt.show()

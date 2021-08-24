# Convert riskFnGauss(Fig 5.13) to python/JAX
# Author: Garvit9000c

import superimport

import numpy as np
import matplotlib.pyplot as plt

for n in [5,20]:
  #Domain
  mus=np.arange(-1.8,1.8,0.1)

  #constants
  M = len(mus)
  n0 = 1
  n0B = 5
  mu0 = 0
  pi=np.pi

  #functions
  r1 = (1/n) * np.ones(M)
  r2 = (pi/(2*n)) * np.ones(M)
  r3 = (mus-mu0)**2
  r4 = (n+n0)**(-2) * (n + n0**2*(mu0-mus)**2)
  r5 = (n+n0B)**(-2) * (n + n0B**2*(mu0-mus)**2)

  #Plotting
  plt.title('risk function for n='+str(n))
  legendStr = ['mle','median','fixed','postmean1', 'postmean5']
  style=[['b','solid'],['r','dotted'],['k','dashdot'],['lime','dashed'],['cyan','solid']]
         
  r = [r1, r2, r3, r4, r5]
  for i in range(len(r)):
    plt.plot(mus,r[i],style[i][0],lineStyle=style[i][1],label=legendStr[i])
  plt.legend(loc='upper left',prop={'size': 7})

  if n==5:
    plt.ylim(0,0.5)
  else:
    plt.ylim(0,0.18)
  plt.xticks([-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])

  #axis Labels
  plt.xlabel(r'$\theta_*$')
  plt.ylabel(r'$R(\theta_*,\delta)$')

  D={5:'a',20:'b'}
  plt.savefig('../figures/riskFnGauss('+D[n]+').pdf', dpi=300)
  plt.show()

# Central Limit theorem

# Author: Animesh Gupta

import superimport

import numpy as np
import matplotlib.pyplot as plt 

#number of samples  
samples = 100000
bins = 20
N = [1, 5] 
  
  
def convolutionHist(N,sampleSize,bins):
# Generating 1, 5 random numbers from 1 to 5 
# taking their mean and appending it to list means. 
    means = []
    for j in N: 
        # Generating seed so that we can get same result  
        # every time the loop is run... 
        np.random.seed(1)
        x = np.mean( 
            np.random.beta( 
                1, 5,[sampleSize,j]),axis=1) 
        means.append(x) 
        
    return means

def plot_convolutionHist(means,N,sampleSize,bins):
    for i, mean_ in zip(N, means):
        counts, nbinslocation = np.histogram(mean_, bins=20)

        counts = counts / (sampleSize/bins)
        
        plt.figure(figsize=(4,4))
        plt.title(f'N = {i}')
        plt.bar(nbinslocation[:-1], counts, width=0.02, color='tab:blue', align='edge')
        plt.xticks(np.linspace(0,1,3))
        plt.yticks(np.linspace(0,3,4))
        plt.xlim(0, 1)
        plt.ylim(0, 3)
        plt.savefig(f"../figures/clt_N_{i}.pdf", dpi=300)
        plt.show()
        
means = convolutionHist(N,samples,bins)
plot_convolutionHist(means,N,samples,bins)

# DNA Sequence Demonstration
# Author: Drishtii@
# Based on
# https://github.com/probml/pmtk3/blob/master/demos/seqlogoDemo.m

#!pip install logomaker

import superimport

import numpy as np
import pandas as pd
import pyprobml_utils as pml
import logomaker
import matplotlib.pyplot as plt

li = ['atagccggtacggca', 'ttagctgcaaccgca', 'tcagccactagagca', 'ataaccgcgaccgca', 'ttagccgctaaggta', 'taagcctcgtacgta', 'ttagccgttacggcc', 'atatccggtacagta', 'atagcaggtaccgaa', 'acatccgtgacggaa']

new_li = []
for i in range(len(li[0])):
    r = ''
    for j in range(len(li)):
        r += li[j][i]
    new_li.append(r)

position_weight_matrix = np.zeros((4, 15))
alphabets = ['a', 'c', 'g', 't']
for seq in range(len(new_li)):
  for alphabet in range(len(alphabets)):
    position_weight_matrix[alphabet][seq] = new_li[seq].count(alphabets[alphabet])/5

df = pd.DataFrame(position_weight_matrix.T, columns = ['A','C','G', 'T'])
df.index = np.arange(1, len(df) + 1)

logos = logomaker.Logo(df)
logos.ax.set_xticks(np.arange(1, 16))
logos.ax.set_yticks(np.arange(3))
logos.ax.set_ylabel('Bits')
logos.ax.set_xlabel('Sequence Position')
pml.savefig('seqlogo.pdf')
plt.show()

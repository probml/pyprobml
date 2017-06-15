#!/usr/bin/env python

# Precision-recall curve for two hypothetical classifications systems.
# A is better than B. 

import matplotlib.pyplot as pl
import numpy as np

fA = np.vectorize(lambda x: 1 - x**3)
fB = np.vectorize(lambda x: 1 - x**(3/2))
x = np.arange(0, 1, 0.01)
pl.plot(x, fA(x), 'r-')
pl.plot(x, fB(x), 'b-')

pl.text(0.6, 0.8, 'A', color='red', size='x-large')
pl.text(0.1, 0.8, 'B', color='blue', size='x-large')

pl.axis([0, 1, 0, 1.01])
pl.xlabel('recall', size='xx-large')
pl.ylabel('precision', size='xx-large')
pl.legend()
pl.savefig('PRhand.png')
pl.show()

#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np

p = np.random.rand(5000, 2) * 4 - 2
inner = np.sum(p ** 2, axis=1) <= 4
pl.figure(figsize=(10, 10))
pl.plot(p[inner, 0], p[inner, 1], 'bo')
pl.plot(p[~inner, 0], p[~inner, 1], 'rD')
pi_estimate = np.sum(inner) / 5000 * 4

print('the estimated pi = %f' % pi_estimate)
print('the standard pi = %f' % np.pi)
err = np.abs(np.pi - pi_estimate) / np.pi
print('err = %f' % err)
pl.savefig('mcEstimatePi.png')
pl.show()

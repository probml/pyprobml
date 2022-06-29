import superimport

import numpy as np
import matplotlib.pyplot as plt


def plotColors():

    colors = ['b', 'r', 'k', 'g', 'c', 'y',
              'm', 'r', 'b', 'k', 'g', 'c', 'y', 'm']
    symbols = ['o', 'x', '*', '>', '<', '^',
               'v', '+', 'p', 'h', 's', 'd', 'o', 'x']
    styles = ['-', ':', '-.', '--', '-', ':', '-.',
              '--', '-', ':', '-.', '--', '-', ':', '-.', '--']
    Str = []
    for i in range(0, len(colors)):
        Str.append(colors[i] + styles[i])
    return [styles, colors, symbols, Str]


def box(u): return (1/2)*(abs(u) <= 1)


def epa(u): return ((3/4)*(1 - np.power(u, 2))*(abs(u) <= 1))


def tri(u): return (70/81)*np.power((1- np.power(abs(u), 3)),3)*(abs(u) <= 1)


def gauss(u): return (1/np.sqrt(2*np.pi)) * np.exp(-np.power(u, 2)/2)


fns = [box, epa, tri, gauss]
names = ['Boxcar', 'Epanechnikov', 'Tricube', 'Gaussian']

xs = np.arange(-1.5, 1.501, 0.01)
[styles, colors, symbols, Str] = plotColors()

smoothingKernalPlot = plt.figure()

for i in range(0, len(fns)):
    f = fns[i]
    fx = f(xs)
    b = xs[1]-xs[0]
    print('integral is '+str(sum(fx)))
    smoothingKernalPlot = plt.plot(xs, fx, styles[i]+colors[i])

smoothingKernalPlot = plt.legend(names)
#plt.savefig('smoothingKernelPlot')
plt.show()

# Author: Shehjad Khan

import superimport

import numpy as np
import os
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fisherDiscrimVowelDemo():
    data = loadmat('../data/vowelTrain')
    Xtrain = data['Xtrain']
    ytrain = data['ytrain']
    N, D = Xtrain.shape
    pca = PCA(min(N, D))
    pca.fit(Xtrain)
    Z = pca.transform(Xtrain)

    C = np.max(ytrain)
    muC = np.zeros((C, D))

    for c in range(0, C):
        muC[c, :] = (np.mean((Xtrain[np.where(ytrain == (c+1))[0], :]), axis=0))

    muC2d = pca.transform(muC)

    symbols = '+ovd*.xs^d><ph'

    for c in range(0, C):
        ndx = np.where(ytrain == (c+1))
        plt.scatter(Z[ndx, 0], Z[ndx, 1], marker=symbols[c])
    plt.savefig('../figures/fisherDiscrimVowelPCA')
    plt.show()


    clf = LinearDiscriminantAnalysis()
    W = clf.fit_transform(Xtrain, ytrain)
    W[:, 1] = - W[:, 1]
    Xlda = Xtrain*W
    mdl = LinearDiscriminantAnalysis()
    model = mdl.fit(Xtrain, ytrain)

    print(muC.shape)
    muC2dlda = np.dot(muC, W)
    for c in range(0, C):
        ndx = np.where(ytrain == c+1)
        plt.scatter(muC2dlda[c, 0], muC2dlda[c, 1],  'o')

    plt.show()

def main():
    fisherDiscrimVowelDemo()

if __name__ == '__main__':
    if os.path.isdir('Scripts'):
        os.chdir('./Scripts')
    main()

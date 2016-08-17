#!/usr/bin/env python

# Pairwise scatterplots of Fisher Iris features. The diagonal plots contain
# the marginal histograms of the features, while the off diagonals contain
# pairs of features.

import matplotlib.pyplot as pl
from itertools import permutations
from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']

feature_combinations = list(permutations(range(4), 2))
for i in range(16):
    if i % 5 == 0:
        features_sel = features[:, int(i/5)]
        pl.subplot(4, 4, i+1)
        pl.hist(features_sel, color='w')
        pl.xlabel(feature_names[int(i/5)], fontsize=10)
        pl.ylabel(feature_names[int(i/5)], fontsize=10)
    else:
        pl.subplot(4, 4, i+1)
        for t, m, c in zip(range(3), 'D*o', 'bgr'):
            feature_chosen = feature_combinations[i-1-(i//5)]
            pl.scatter(features[target == t, feature_chosen[0]],
                       features[target == t, feature_chosen[1]],
                       marker=m, color=c)
        pl.xlabel(feature_names[feature_chosen[0]], fontsize=10)
        pl.ylabel(feature_names[feature_chosen[1]], fontsize=10)
    pl.xticks(())
    pl.yticks(())

pl.savefig('fisheririsDemo.png')
pl.show()

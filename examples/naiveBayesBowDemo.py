#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np
import utils.util as util
from sklearn.naive_bayes import MultinomialNB

data = util.load_mat('XwindowsDocData')
xtrain = data['xtrain']
ytrain = data['ytrain']

clf = MultinomialNB()
clf.fit(xtrain, ytrain.ravel())
counts = clf.feature_count_
y_counts = clf.class_count_
for i in range(len(counts)):
    pl.figure()
    pl.bar(np.arange(len(counts[i])), counts[i] / y_counts[i])
    pl.title('p(xj=1|y=%d)' % (i + 1))
    pl.savefig('naiveBayesBowDemo_%d.png' % i)
pl.show()

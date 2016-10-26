#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np
import utils.util as util
from sklearn.cluster import KMeans

data = util.load_mat('heightWeight')
data = data['heightWeightData']
markers = 'Dox'
colors = 'rgb'

for i in range(3):
    KM_model = KMeans(init='k-means++', n_clusters=i+1)
    labels = KM_model.fit_predict(data[:, [1, 2]])
    labels_unique = np.unique(labels)
    fig = pl.figure(i)
    for j in range(len(labels_unique)):
        data_chosen = data[labels == labels_unique[j]]
        pl.scatter(data_chosen[:, 1], data_chosen[:, 2],
                   marker=markers[j],
                   color=colors[j])
    pl.title('k = %s' % (i+1))
    pl.savefig('kmeansHeightWeight_%s.png' % (i+1))
pl.show()

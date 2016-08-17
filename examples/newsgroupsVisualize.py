#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
import utils.util as util
from scipy import ndimage

data = util.load_mat('20news_w100')
documents = data['documents']
documents = documents.toarray().T
newsgroups = data['newsgroups'][0]

#sort documents by number of words and choose the first 1000
chosen_docs_arg = np.argsort(np.sum(documents, axis=1))
chosen_docs_arg = chosen_docs_arg[-1000:][::-1]  # descend
documents = documents[chosen_docs_arg]
newsgroups = newsgroups[chosen_docs_arg]

#sort by newsgroups label
sorted_arg = np.argsort(newsgroups)
documents = documents[sorted_arg]
newsgroups = newsgroups[sorted_arg]

#zoom the image to show it
image = ndimage.zoom(documents, (1, 10))
pl.imshow(image, cmap=pl.cm.gray, interpolation='none')
#draw a red line betweent different newsgroups
groups_label = np.unique(newsgroups)
for i in range(len(groups_label) - 1):
    y, = np.where(newsgroups == groups_label[i + 1])
    y = y[0]
    pl.plot([y]*newsgroups.shape[0], 'r', lw=2)
pl.axis('tight')
pl.xlabel('words')
pl.ylabel('documents')
pl.xticks(range(0, 1001, 100), range(0, 101, 10))
pl.yticks(range(0, 1001, 100), range(0, 1001, 100))
pl.savefig('newsgroupsVisualize.png')
pl.show()

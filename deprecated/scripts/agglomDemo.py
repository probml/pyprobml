# Agglomerative Clustering Demo
# Author: Animesh Gupta

import superimport

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pyprobml_utils as pml

X = np.array([[1,2],
    [2.5,4.5],
    [2,2],
    [4,1.5],
    [4,2.5],])

labels = range(1, 6)
plt.figure(figsize=(10, 6))
plt.yticks(np.linspace(0,5,11))
plt.ylim(0,5)
plt.grid(color='gray', linestyle='dashed')
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom', fontsize=25, color="red")
pml.savefig("agglom_demo_data.pdf", dpi=300)

linked = linkage(X, 'single')

labelList = range(1, 6)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
pml.savefig("agglom_demo_dendrogram.pdf", dpi=300)

plt.show()
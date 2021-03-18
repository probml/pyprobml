import scipy.io
import os
from scipy.spatial.distance import pdist
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import numpy as np
import seaborn


if os.path.isdir('scripts'):
    os.chdir('scripts')
data = scipy.io.loadmat('../data/yeastData310.mat')

X = data['X']

corrDist = pdist(X, 'correlation')
clusterTree = AgglomerativeClustering(n_clusters=16, linkage="average")
clusterTree.fit(X)
clusters = clusterTree.labels_

clusterYeastHier16, axes = plt.subplots(4, 4)
clusterYeastHier16 = clusterYeastHier16.suptitle('Hierarchical Clustering of Profiles ', y = 1 ,fontsize = 20)
times = data['times'].reshape(7, )
for c in range(0, 16):
    occurences = np.argwhere(clusters == (c))
    row = c//4
    col = c%4
    for occ in occurences:
        axes[row][col].plot(times, X[occ, :].reshape(7,))
    
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('../figures/clusterYeastHier16')
plt.show()

clusterYeastRowPerm = seaborn.clustermap(X[:, 1:])
plt.title('hierarchical clustering')
plt.savefig('../figures/clusterYeastRowPerm')
plt.show()

Z = scipy.cluster.hierarchy.linkage(corrDist, 'average')
scipy.cluster.hierarchy.dendrogram(Z, truncate_mode='lastp')
plt.title('average link')
plt.tick_params(labelbottom=False)
plt.savefig('../figures/clusterYeastAvgLink')
plt.show()

Z = scipy.cluster.hierarchy.linkage(corrDist, 'complete')
scipy.cluster.hierarchy.dendrogram(Z, truncate_mode='lastp')
plt.title('complete link')
plt.tick_params(labelbottom=False)
plt.savefig('../figures/clusterYeastCompleteLink')
plt.show()

Z = scipy.cluster.hierarchy.linkage(corrDist, 'single')
scipy.cluster.hierarchy.dendrogram(Z, truncate_mode='lastp')
plt.title('single link')
plt.tick_params(labelbottom=False)
plt.savefig('../figures/clusterYeastSingleLink')
plt.show()
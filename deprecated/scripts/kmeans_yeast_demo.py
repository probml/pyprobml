import superimport

from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import requests
from io import BytesIO

url = 'https://github.com/probml/probml-data/blob/main/data/yeastData310.mat?raw=true'
response = requests.get(url)
rawdata = BytesIO(response.content)
data = loadmat(rawdata) # dictionary containing 'X', 'genes', 'times'

X = data['X']

# Cluster yeast data using Kmeans

kmeans = KMeans(n_clusters=16, random_state=0, algorithm='full').fit(X)
times = data['times']
X = X.transpose()

labels = kmeans.labels_
clu_cen = kmeans.cluster_centers_

clusters = [[] for i in range(0, 16)]

for (i, l) in enumerate(labels):
    clusters[l].append(i)

times = times.reshape((7,))

# Visualizing all the time series assigned to each cluster

plt.figure()
for l in range(0, 16):
    plt.subplot(4, 4, l + 1)
    if clusters[l] != []:
        plt.plot(times, X[:, clusters[l]])
plt.suptitle("K-Means Clustering of Profiles")
pml.savefig("yeastKmeans16.pdf")
plt.show()

# Visualizing the 16 cluster centers as prototypical time series.

plt.figure()
for l in range(0, 16):
    plt.subplot(4, 4, l + 1).axis('off')
    plt.plot(times, clu_cen[l, :])
plt.suptitle("K-Means centroids")
pml.savefig("clusterYeastKmeansCentroids16.pdf")
plt.show()


import superimport

from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

data = loadmat('/pyprobml/data/yeastData310.mat') #  dictionary containing 'X', 'genes', 'times'
X = data['X']

# Cluster yeast data using Kmeans

kmeans = KMeans(n_clusters=16,random_state=0,algorithm='full').fit(X)
times = data['times']
X = X.transpose()
  
labels = kmeans.labels_
clu_cen = kmeans.cluster_centers_

clusters = [[] for i in range(0,16)]

for (i,l) in enumerate(labels):
  clusters[l].append(i)

times = times.reshape((7,))
  
# Visualizing all the time series assigned to each cluster

for l in range(0,16):
  plt.subplot(4,4,l+1)
  if clusters[l] != []:
    plt.plot(times,X[:,clusters[l]])
    
plt.suptitle("K-Means Clustering of Profiles")
plt.savefig("/pyprobml/figures/yeastKmeans16.pdf",  dpi=300)
plt.show()

# Visualizing the 16 cluster centers as prototypical time series.

for l in range(0,16):
  plt.subplot(4,4,l+1).axis('off')
  plt.plot(times,clu_cen[l,:])
    
plt.suptitle("K-Means centroids")
plt.savefig("/pyprobml/figures/clusterYeastKmeansCentroids16.pdf",  dpi=300)
plt.show()


# yeast gene expression data plotted as a time series
plt.plot(times,X,'o-')
plt.title('yeast microarray data') 
plt.xlabel('time') 
plt.ylabel('genes')
plt.xlim([0,max(times)])
plt.xticks(ticks=times,labels=times)
plt.savefig("/pyprobml/figures/yeastTimeSeries.pdf",  dpi=300)
plt.show()

# yeast gene expression data plotted as a heat map
basic_cols=['#66ff00', '#000000', '#FF0000'] # green-black-red
my_cmap=LinearSegmentedColormap.from_list('mycmap', basic_cols)
plt.xticks(ticks=[i+0.5 for i in range(0,7)],labels=times)
plt.pcolormesh(X.transpose(),cmap=my_cmap)
plt.title('yeast microarray data') 
plt.xlabel('time') 
plt.ylabel('genes')
plt.colorbar()
plt.savefig("/pyprobml/figures/yeastHeatMap.pdf",  dpi=300)

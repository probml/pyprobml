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

times = data['times']
X = X.transpose()
times = times.reshape((7,))


# yeast gene expression data plotted as a time series
plt.figure()
plt.plot(times, X, 'o-')
plt.title('yeast microarray data')
plt.xlabel('time')
plt.ylabel('genes')
plt.xlim([0, max(times)])
plt.xticks(ticks=times, labels=times)
pml.savefig("yeastTimeSeries.pdf")
plt.show()

# yeast gene expression data plotted as a heat map
plt.figure()
basic_cols = ['#66ff00', '#000000', '#FF0000']  # green-black-red
my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
plt.xticks(ticks=[i + 0.5 for i in range(0, 7)], labels=times)
plt.pcolormesh(X.transpose(), cmap=my_cmap)
plt.title('yeast microarray data')
plt.xlabel('time')
plt.ylabel('genes')
plt.colorbar()
pml.savefig("yeastHeatMap.pdf")
plt.show()


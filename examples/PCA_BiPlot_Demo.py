from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram

import os
fig_folder = os.getcwd()

#Pull in iris dataset.
iris_dataset_url = 'https://raw.githubusercontent.com/pydata/pandas/master/pandas/tests/data/iris.csv'
iris = pd.read_csv(iris_dataset_url)

print(iris.head())
print(iris.corr().round(2))

#Feature names
ColNs = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

X = iris[ColNs]
X_sd = iris[ColNs]

#Standardize X_sd
for n in ColNs:
    val = (X[n] - np.mean(X[n])) / np.std(X[n])
    X_sd.loc[:, n] = val

#Determine PCA directions
pca = PCA(n_components=4).fit(X_sd)
Dirs = pca.components_

#Create summary output.
CompNs = ["Comp " + str(i) for i in range(1, 5)]
IdxNs = ['Standard Deviation', 'Proportion of Variance', 'Cumulative Proportion']
Summary = pd.DataFrame(index=IdxNs, columns=CompNs)
ProjectedData = np.dot(X_sd, np.transpose(Dirs))
Sds = np.std(ProjectedData, 0)
Vars = Sds ** 2
Summary.loc['Standard Deviation', :] = Sds
Summary.loc['Proportion of Variance', :] = Vars / sum(Vars)
Summary.loc['Cumulative Proportion', :] = np.cumsum(Summary.loc['Proportion of Variance', :])

print(Summary)

#Graph of cumulative variance explained
fig1 = plt.figure(1)
plt.plot(Vars)
plt.xticks(range(4), CompNs)
plt.draw()

#2D Biplot
Ns = np.unique(iris['Name'])
Cs = ['blue', 'red', 'green']
print('names')
print(Ns)
print('colors')
print(Cs)

fig2, ax = plt.subplots(num=2)

for i in range(4):
    y1 = Dirs[0, i]*3
    y2 = Dirs[1, i]*3
    ax.arrow(0, 0, y1, y2, head_width=0.05)
    ax.annotate(ColNs[i], (y1, y2))

for i in range(3):
    sel = list(iris.loc[iris['Name'] == Ns[i]].index)
    m = ProjectedData[sel, :]
    ax.scatter(m[:, 0], m[:, 1], c=Cs[i], s=100)

ax.set_xlabel('Comp 1')
ax.set_ylabel('Comp 2')
plt.draw()
plt.savefig(os.path.join(fig_folder, 'pcaIris2dBiplotPython.pdf'))



#3D Biplot
fig3 = plt.figure(3)
ax = fig3.add_subplot(111, projection='3d')

for i in range(3):
    sel = [x for x in range(150) if iris.loc[x, 'Name'] == Ns[i]]
    m = ProjectedData[sel, :]
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], c=Cs[i], s=100)
    ax.set_xlabel('Comp 1')
    ax.set_ylabel('Comp 2')
    ax.set_zlabel('Comp 3')
for i in range(4):
    x = Dirs[0, i]*3
    y = Dirs[1, i]*3
    z = Dirs[2, i]*3
    ax.plot([0, x], [0, y], [0, z], c='black')
    ax.text(x, y, z, ColNs[i], color='black', size=20)
plt.draw()
plt.savefig(os.path.join(fig_folder, 'pcaIris3dBiplotPython.pdf'))

#KMeans clustering (applied to unstandardized data)
KM3 = KMeans(n_clusters=3)
KM3.fit(X)
labels = KM3.labels_
iris['KMeans'] = labels

#3D plot of PCA-projected points classified according to KMeans
fig4 = plt.figure(4)
ax = fig4.add_subplot(111, projection='3d')
for i in range(3):
    sel = [x for x in range(150) if labels[x] == i]
    m = ProjectedData[sel, :]
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], c=Cs[i], s=100)
    ax.set_xlabel('Comp 1')
    ax.set_ylabel('Comp 2')
    ax.set_zlabel('Comp 3')
ax.set_title('Clusters according to K-Means')
plt.draw()

#Hierarchical clustering (applied to unstandardized data)
linkage_matrix = linkage(X, "ward")

#3D plot of PCA-projected points classified according to hierarchical clustering
fig5 = plt.figure(5)
ddata = dendrogram(linkage_matrix)
plt.draw()

iris['Ward'] = fcluster(linkage_matrix, 9, 'distance')

#Function to replicate R's table function
def table(x, y):
    xuni = np.unique(x)
    yuni = np.unique(y)
    res = pd.DataFrame(index=xuni, columns=yuni)
    for x1 in xuni:
        for y1 in yuni:
            res.loc[x1, y1] = sum((x == x1) & (y == y1))
    return res

#Print tables to inspect how well classification has worked.
print('Kmeans purity matrix')
print(table(iris['KMeans'], iris['Name']))

print('Ward purity matrix')
print(table(iris['Ward'], iris['Name']))

plt.show(block=True)

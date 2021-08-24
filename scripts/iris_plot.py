import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
import pyprobml_utils as pml
import seaborn as sns;
sns.set(style="ticks", color_codes=True)

import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows

import sklearn
from sklearn.datasets import load_iris
iris = load_iris()

# Extract numpy arrays
X = iris.data 
y = iris.target

# Convert to pandas dataframe 
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['label'] = pd.Series(iris.target_names[y], dtype='category')

# we pick a color map to match that used by decision tree graphviz 
#cmap = ListedColormap(['#fafab0','#a0faa0', '#9898ff']) # orange, green, blue/purple
#cmap = ListedColormap(['orange', 'green', 'purple']) 
palette = {'setosa': 'orange', 'versicolor': 'green', 'virginica': 'purple'}

g = sns.pairplot(df, vars = df.columns[0:4], hue="label", palette=palette)
#g = sns.pairplot(df, vars = df.columns[0:4], hue="label")
pml.savefig('iris_scatterplot_purple.pdf')
plt.show()


# Change colum names
iris_df = df.copy()
iris_df.columns =  ['sl', 'sw', 'pl', 'pw'] + ['label'] 

g = sns.pairplot(iris_df, vars = iris_df.columns[0:4], hue="label")
plt.tight_layout()
pml.savefig('iris_pairplot.pdf')
plt.show()


sns.stripplot(x="label", y="sl", data=iris_df, jitter=True)
pml.savefig('iris_sepal_length_strip_plot.pdf')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import os

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
df = pd.DataFrame(data=X, columns=['sl', 'sw', 'pl', 'pw'])
# create column for labels
df['label'] = pd.Series(iris.target_names[y], dtype='category')


# 2d scatterplot
#https://seaborn.pydata.org/generated/seaborn.pairplot.html

# Make a dataframe with nicer labels for printing
#iris_df = sns.load_dataset("iris")
iris_df = df.copy()
iris_df.columns = iris['feature_names'] + ['label'] 


g = sns.pairplot(iris_df, vars = iris_df.columns[0:4], hue="label")
plt.tight_layout()
plt.savefig("../figures/iris_pairplot.pdf")
plt.show()

if 0:
    # pick ugly colors to match iris_dtree.py
    palette = {'setosa': 'red', 'versicolor': 'yellow', 'virginica': 'blue' }
    g = sns.pairplot(iris_df, vars = iris_df.columns[0:4] ,
                     hue="label", palette=palette)
    plt.savefig("../figures/iris_pairplot-ryb.pdf")
    plt.show()

sns.stripplot(x="label", y="sl", data=df, jitter=True)
plt.savefig('../figures/iris_sepal_length_strip_plot.pdf', dpi=300);
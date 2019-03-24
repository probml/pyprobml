import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


from sklearn.datasets import load_iris
iris = load_iris()
# show attributes
dir(iris)
# ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']

# Extract numpy arrays
X = iris.data 
y = iris.target
print(np.shape(X)) # (150, 4)
print(np.c_[X[0:3,:], y[0:3]]) # concatenate columns
#[[5.1 3.5 1.4 0.2 0. ]
# [4.9 3.  1.4 0.2 0. ]
# [4.7 3.2 1.3 0.2 0. ]]

# The data is sorted by class. Let's shuffle the rows.
N = np.shape(X)[0]
rng = np.random.RandomState(42)
perm = rng.permutation(N)
X = X[perm]
y = y[perm]
print(np.c_[X[0:3,:], y[0:3]])
#[[6.1 2.8 4.7 1.2 1. ]
# [5.7 3.8 1.7 0.3 0. ]
# [7.7 2.6 6.9 2.3 2. ]]

# Convert to pandas dataframe
import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows
 
df = pd.DataFrame(data=X, columns=['sl', 'sw', 'pl', 'pw'])
# create column for labels
s = pd.Series(iris.target_names[y], dtype='category')
df['label'] = s


# Summary statistics
df.describe(include='all')

# Peak at the data
df.head()

# Create latex table from first 5 rows 
str = df[:6].to_latex(index=False, escape=False)

# 2d scatterplot
#https://seaborn.pydata.org/generated/seaborn.pairplot.html
import seaborn as sns;
sns.set(style="ticks", color_codes=True)
# Make a dataframe with nicer labels for printing
#iris_df = sns.load_dataset("iris")
iris_df = df.copy()
iris_df.columns = iris['feature_names'] + ['label'] 
g = sns.pairplot(iris_df, vars = iris_df.columns[0:3] , hue="label")
save_fig("iris-scatterplot.pdf")
plt.show()


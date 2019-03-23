# https://www.kaggle.com/devanshbesain/exploration-and-analysis-auto-mpg
#https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_regression.ipynb


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150) # wide windows

import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
 
  
#from sklearn.datasets import fetch_openml
#auto = fetch_openml('autoMpg', cache=True)
# The OpenML version converts the original categorical data
# to integers starting at 0.
# We want the 'raw' data.


#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# We made a cached copy since UCI repository is often down
url = 'https://raw.githubusercontent.com/probml/pyprobml/master/data/mpg.csv'
#column_names = ['mpg','cylinders','displacement','horsepower','weight',
#                'acceleration', 'model_year', 'origin', 'name'] 
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Year', 'Origin', 'Name']
df = pd.read_csv(url, names=column_names, sep='\s+', na_values="?")

# The last column (name) is a unique id for the car, so we drop it
df = df.drop(columns=['Name'])


df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 398 entries, 0 to 397
Data columns (total 8 columns):
MPG             398 non-null float64
Cylinders       398 non-null int64
Displacement    398 non-null float64
Horsepower      392 non-null float64
Weight          398 non-null float64
Acceleration    398 non-null float64
Year            398 non-null int64
Origin          398 non-null int64
dtypes: float64(5), int64(3)
memory usage: 25.0 KB
"""

# We notice that there are only 392 horsepower rows, but 398 of the others.
# This is because the HP column has 6 missing values (also called NA, or
# not available).
# There are 3 main ways to deal with this:
# Drop the rows with any missing values using dropna()
# Drop any columns with any missing values using drop()
# Replace the missing vales with some other valye (eg the median) using fillna.
# (This latter is called missing value imputation.)

df = df.dropna()

# Origin is categorical (1=USA, 2=Europe, 3=Japan)
df['Origin'] = df.Origin.replace([1,2,3],['USA','Europe','Japan'])
df['Origin'] = df['Origin'].astype('category')
# Cylinders is an integer in [3,4,5,6,8]
#df['Cylinders'] = df['Cylinders'].astype('category')
# Year is an integer year (between 70 and 82)
#df['Year'] = df['Year'].astype('category')
df0 = df.copy()

# Let us check the datatypes
print(df.dtypes)
"""
MPG              float64
Cylinders          int64
Displacement     float64
Horsepower       float64
Weight           float64
Acceleration     float64
Year               int64
Origin          category
"""
# Let us check the categories
df['Origin'].cat.categories #Index(['Europe', 'Japan', 'USA'], dtype='object')

# Let us inspect the data
df.tail()

"""
      MPG Cylinders  Displacement  Horsepower  Weight  Acceleration Year  Origin
393  27.0         4         140.0        86.0  2790.0          15.6   82     USA
394  44.0         4          97.0        52.0  2130.0          24.6   82  Europe
395  32.0         4         135.0        84.0  2295.0          11.6   82     USA
396  28.0         4         120.0        79.0  2625.0          18.6   82     USA
397  31.0         4         119.0        82.0  2720.0          19.4   82     USA
"""

#https://www.kaggle.com/devanshbesain/exploration-and-analysis-auto-mpg

# Plot mpg distribution for cars from different countries of origin
data = pd.concat( [df['MPG'], df['Origin']], axis=1)
fig, ax = plt.subplots()
ax = sns.boxplot(x='Origin', y='MPG', data=data)
ax.axhline(data.MPG.mean(), color='r', linestyle='dashed', linewidth=2)
plt.savefig(os.path.join(figdir, 'auto-mpg-origin-boxplot.pdf'))
plt.show()

# Plot mpg distribution for cars from different years
data = pd.concat( [df['MPG'], df['Year']], axis=1)
fig, ax = plt.subplots()
ax = sns.boxplot(x='Year', y='MPG', data=data)
ax.axhline(data.MPG.mean(), color='r', linestyle='dashed', linewidth=2)
plt.savefig(os.path.join(figdir, 'auto-mpg-year-boxplot.pdf'))
plt.show()




# Convert origin string (factor) to integer
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
origin = df['Origin']
origin = encoder.fit_transform(origin)
# Check the data
print(np.unique(origin)) # [0 1 2] # Note the same as original [1,2,3]
# Check the encoding - happens to be the same as original ordering
print(encoder.classes_) # ['Europe' 'Japan' 'USA'] 
# Convert back (from printing purposes)
origin_names = encoder.inverse_transform(origin)
assert np.array_equal(origin_names, df['Origin'])

# Convert integer encoding to one-hot vectors
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
origin = origin.reshape(-1, 1) # Convert (N) to (N,1)
origin_onehot = encoder.fit_transform(origin) # Sparse array
# Convert to dense array for printing purposes
print(origin_onehot[-5:,:].toarray())
"""
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
"""

"""
# We shoukd be able to combine LabelEncoder and OneHotEncoder together
# using a Pipeline. However this fails due to known bug: https://github.com/scikit-learn/scikit-learn/issues/3956
# TypeError: fit_transform() takes 2 positional arguments but 3 were given

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('str2int', LabelEncoder()),
    ('int2onehot', OneHotEncoder())
])
origin_onehot2 = pipeline.fit_transform(df['Origin'])
"""


# Convert origin string to one-hot encoding
# New feature for sckit v0.20
# https://jorisvandenbossche.github.io/blog/2017/11/20/categorical-encoder/
# https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696

from sklearn.preprocessing import OneHotEncoder
def one_hot_encode_dataframe_col(df, colname):
  encoder = OneHotEncoder(sparse=False)
  data = df[[colname]] # Extract column as (N,1) matrix
  data_onehot = encoder.fit_transform(data)
  df = df.drop(columns=[colname])
  ncats = np.size(encoder.categories_)
  for c in range(ncats):
    colname_c = '{}:{}'.format(colname, c)
    df[colname_c] = data_onehot[:,c]
  return df, encoder

df_onehot, encoder_origin = one_hot_encode_dataframe_col(df, 'Origin')
df_onehot.tail()
"""
   Cylinders  Displacement  Horsepower  Weight  Acceleration Year  Origin:0  Origin:1  Origin:2
393         4         140.0        86.0  2790.0          15.6   82       0.0       0.0       1.0
394         4          97.0        52.0  2130.0          24.6   82       1.0       0.0       0.0
395         4         135.0        84.0  2295.0          11.6   82       0.0       0.0       1.0
396         4         120.0        79.0  2625.0          18.6   82       0.0       0.0       1.0
397         4         119.0        82.0  2720.0          19.4   82       0.0       0.0       1.0
"""

# See also sklearn-pandas library
#https://github.com/scikit-learn-contrib/sklearn-pandas#transformation-mapping

# Replace year with decade (70s and 80s)
year = df.pop('Year')
decade = [ 70 if (y>=70 and y<=79) else 80 for y in year ]
df['Decade'] =  pd.Series(decade, dtype='category')

#
# Make feature cross between #decades and origin (2*3 values)
import patsy
y = df.pop("MPG") # Remove target column from dataframe and store
df.columns = ['Cyl', 'Dsp','HP', 'Wgt', 'Acc',  'Org', 'Dec'] # Shorten names
df['Org'] = df['Org'].replace(['USA','Europe','Japan'], ['U','J','E'])
df_cross = patsy.dmatrix('Dec:Org + Cyl + Dsp + HP + Wgt + Acc', df, return_type='dataframe')
df_cross.tail()
"""
      Intercept  Org[T.J]  Org[T.U]  Dec[T.80]:Org[E]  Dec[T.80]:Org[J]  Dec[T.80]:Org[U]  Cyl    Dsp     HP     Wgt   Acc
387        1.0       0.0       1.0               0.0               0.0               1.0  6.0  262.0   85.0  3015.0  17.0
388        1.0       0.0       1.0               0.0               0.0               1.0  4.0  156.0   92.0  2585.0  14.5
389        1.0       0.0       1.0               0.0               0.0               1.0  6.0  232.0  112.0  2835.0  14.7
390        1.0       0.0       0.0               1.0               0.0               0.0  4.0  144.0   96.0  2665.0  13.9
391        1.0       0.0       1.0               0.0               0.0               1.0  4.0  135.0   84.0  2370.0  13.0
"""
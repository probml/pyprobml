# Exploratory data analysis for auto-mpg dataset
# https://www.kaggle.com/devanshbesain/exploration-and-analysis-auto-mpg
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_regression.ipynb


import superimport

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('precision', 2)  # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)  # wide windows

figdir = "../figures"



#from sklearn.datasets import fetch_openml
#auto = fetch_openml('autoMpg', cache=True)
# The OpenML version converts the original categorical data
# to integers starting at 0.
# We want the 'raw' data.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# We made a cached copy since UCI repository is often down
#url = 'https://raw.githubusercontent.com/probml/pyprobml/master/data/mpg.csv'
# column_names = ['mpg','cylinders','displacement','horsepower','weight',
#                'acceleration', 'model_year', 'origin', 'name']
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Year', 'Origin', 'Name']
df = pd.read_csv(url, names=column_names, sep='\s+', na_values="?")

# The last column (name) is a unique id for the car, so we drop it
df = df.drop(columns=['Name'])


# df.info()


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
df['Origin'] = df.Origin.replace([1, 2, 3], ['USA', 'Europe', 'Japan'])
df['Origin'] = df['Origin'].astype('category')
# Cylinders is an integer in [3,4,5,6,8]
#df['Cylinders'] = df['Cylinders'].astype('category')
# Year is an integer year (between 70 and 82)
#df['Year'] = df['Year'].astype('category')
df0 = df.copy()

# Let us check the datatypes
# print(df.dtypes)

# Let us check the categories
# df['Origin'].cat.categories

# Let us inspect the data
# df.tail()

# https://www.kaggle.com/devanshbesain/exploration-and-analysis-auto-mpg

# Plot mpg distribution for cars from different countries of origin
data = pd.concat([df['MPG'], df['Origin']], axis=1)
fig, ax = plt.subplots()
ax = sns.boxplot(x='Origin', y='MPG', data=data)
ax.axhline(data.MPG.mean(), color='r', linestyle='dashed', linewidth=2)
plt.savefig(os.path.join(figdir, 'auto-mpg-origin-boxplot.pdf'))
plt.show()

# Plot mpg distribution for cars from different years
data = pd.concat([df['MPG'], df['Year']], axis=1)
fig, ax = plt.subplots()
ax = sns.boxplot(x='Year', y='MPG', data=data)
ax.axhline(data.MPG.mean(), color='r', linestyle='dashed', linewidth=2)
plt.savefig(os.path.join(figdir, 'auto-mpg-year-boxplot.pdf'))
plt.show()

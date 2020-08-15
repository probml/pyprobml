# Exploratory data analysis for auto-mpg dataset
# https://www.kaggle.com/devanshbesain/exploration-and-analysis-auto-mpg
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_regression.ipynb

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# params: display of data frames
pd.set_option('precision', 2)  # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)  # wide windows

# consistency check for figures dir
figdir = "../figures"
if not os.path.exists(figdir):
    os.makedirs(figdir)

# get data from UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# We made a cached copy since UCI repository is often down
# url = 'https://raw.githubusercontent.com/probml/pyprobml/master/data/mpg.csv'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Year', 'Origin', 'Name']
df = pd.read_csv(url, names=column_names, sep=r'\s+', na_values="?")

# The last column (name) is a unique id for the car, so we drop it
df = df.drop(columns=['Name'])

# Handling NaNs:
# We notice that there are only 392 horsepower rows, but 398 of the others.
# This is because the HP column has 6 missing values (also called NA, or
# not available).
# There are 3 main ways to deal with this:
# Drop the rows with any missing values using dropna()
# Drop any columns with any missing values using drop()
# Replace the missing values with some other value (eg the median) using fillna.
# (This latter is called missing value imputation.)
df = df.dropna()

# Categorical encoding for 'Origin': (1=USA, 2=Europe, 3=Japan)
df['Origin'] = df.Origin.replace([1, 2, 3], ['USA', 'Europe', 'Japan'])
df['Origin'] = df['Origin'].astype('category')

# Let us check the datatypes
print(df.dtypes)

# Plot mpg distribution for cars from different countries of origin
data_origin = pd.concat([df['MPG'], df['Origin']], axis=1)
fig1, ax1 = plt.subplots()
marker_outlier_params = dict(markerfacecolor='r', marker='o', alpha=0.25)
sns.boxplot(x='Origin', y='MPG', data=data_origin, ax=ax1,
            **{'flierprops': marker_outlier_params})
ax1.axhline(data_origin.MPG.mean(), color='r', linestyle='dashed', linewidth=2)
plt.savefig(os.path.join(figdir, 'auto-mpg-origin-boxplot.pdf'))
plt.show()

# Plot mpg distribution for cars from different years
data_year = pd.concat([df['MPG'], df['Year']], axis=1)
fig2, ax2 = plt.subplots()
sns.boxplot(x='Year', y='MPG', data=data_year, ax=ax2,
            **{'flierprops': marker_outlier_params})
ax2.axhline(data_year.MPG.mean(), color='r', linestyle='dashed', linewidth=2)
plt.savefig(os.path.join(figdir, 'auto-mpg-year-boxplot.pdf'))
plt.show()

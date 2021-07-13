
# Datasaurus dataset
# Author: Drishtii

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pyprobml_utils as pml

url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/DatasaurusDozen.tsv'
df = pd.read_csv(url, sep='\t' )

df1 = df[df['dataset'] == 'dino']

df2 = df[df['dataset'] != 'dino']

sns.lmplot(x="x", y="y", col="dataset", fit_reg=False, hue="dataset", data=df1, col_wrap=4, ci=0, height=4)
pml.savefig("dino.pdf")
sns.lmplot(x="x", y="y", col="dataset", fit_reg=False, hue="dataset", data=df2, col_wrap=4, ci=0, height=4)
pml.savefig("datasaurus.pdf")

# Stats of all 12 datasets
datasets = df.groupby('dataset')
datasets.agg(['count', 'mean', 'var'])
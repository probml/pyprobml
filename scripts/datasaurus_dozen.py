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

# Plotting separately:

sns.lmplot(x="x", y="y", col="dataset", fit_reg=False, hue="dataset", data=df1, col_order = ['dino', ' ', ' ', ' '], col_wrap=4, ci=0, height=4) # 'away', 'h_lines', 'v_lines', 'x_shape', 'star', 'hlines', 'dots', 'circle', 'bullseye', 'slant_up', 'slant_down', 'wide_lines'
pml.savefig("dino.pdf")
sns.lmplot(x="x", y="y", col="dataset", fit_reg=False, hue="dataset", data=df2, col_wrap=4, ci=0, height=4)
pml.savefig("datasaurus.pdf")

# Plotting altogether:

sns.lmplot(x="x", y="y", col="dataset", fit_reg=False, hue="dataset", data=df, col_order = ['dino', 'None ', 'None', 'None', 'away', 'h_lines', 'v_lines', 'x_shape', 'star', 'high_lines', 'dots', 'circle', 'bullseye', 'slant_up', 'slant_down', 'wide_lines'], col_wrap=4, ci=0, height=4) # 
pml.savefig("datasaurus_together.pdf")

# Stats of all 12 datasets
datasets = df.groupby('dataset')
datasets.agg(['count', 'mean', 'var'])

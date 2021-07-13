# Datasaurus dataset
# Author: Drishtii

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pyprobml_utils as pml


url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/DatasaurusDozen.tsv'
df = pd.read_csv(url, sep='\t' )

sns.lmplot(x="x", y="y", col="dataset", fit_reg=False, hue="dataset", data=df, col_order = ['dino', 'None ', 'None', 'None', 'away', 'h_lines', 'v_lines', 'x_shape', 'star', 'high_lines', 'dots', 'circle', 'bullseye', 'slant_up', 'slant_down', 'wide_lines'], col_wrap=4, ci=0, height=4)  
pml.savefig("datasaurus_new.pdf")

# Stats of all 12 datasets
datasets = df.groupby('dataset')
datasets.agg(['count', 'mean', 'var'])
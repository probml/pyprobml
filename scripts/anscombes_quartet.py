# Anscombe's quartet 
# Author: Drishtii

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pyprobml_utils as pml

# Anscombe's quartet plot
sns.set_theme(style="ticks")
df = sns.load_dataset("anscombe")
g = sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df, col_wrap=2, ci=None, palette="muted", height=4, scatter_kws={"s": 50, "alpha": 1}, legend_out=True, truncate=False)
g.set(xlim=(2.5, 20.5 ))
pml.savefig("anscombes_quartet")

# Stats of all 4 datasets:
df[['x', 'y']].agg(['count', 'mean', 'var'])


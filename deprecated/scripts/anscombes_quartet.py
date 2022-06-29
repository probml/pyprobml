# Anscombe's quartet 
# Author: Drishtii

import superimport

import seaborn as sns
import matplotlib.pyplot as plt
import pyprobml_utils as pml

sns.set_theme(style="ticks")
df = sns.load_dataset("anscombe")
g = sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df, col_wrap=4, ci=None, palette="muted",
               height=4, scatter_kws={"s": 50, "alpha": 1}, legend_out=True, truncate=False)
g.set(xlim=(2.5, 20.5 ))
pml.savefig("anscombes_quartet.pdf")

names = df['dataset'].unique()
for name in names:
    print(name)
    ndx = df['dataset']==name
    df2 = df[ndx]
    lm = sns.lmplot(x="x", y="y", data=df2, ci=None, truncate=False)
    ax = plt.gca()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    mx = df2['x'].to_numpy().mean();
    my = df2['y'].to_numpy().mean()
    ax.set_title(f'{name}, mx={mx:0.3f}, my={my:0.3f}', fontsize=12)
    print(df2[['x', 'y']].agg(['count', 'mean', 'var']))
    pml.savefig(f"anscombes_quartet_{name}.pdf")


# Compare the two different estimators for the variance
# https://github.com/probml/pml-book/issues/264
for d in ['I', 'II', 'III', 'IV']:
    print('dataset ', d)

    x = df[df['dataset'] == d]['x'].to_numpy()
    print('var x, MLE = {:.2f}'.format(((x - x.mean()) ** 2).mean()))
    print('var x, numpy: {:.2f}'.format(x.var()))
    print('var x, unbiased estimator: {:.2f}\n'.format(x.var(ddof=1)))

    y = df[df['dataset'] == d]['y'].to_numpy()
    print('var y, MLE = {:.2f}'.format(((y - y.mean())**2).mean()))
    print('var y, numpy: {:.2f}'.format(y.var()))
    print('var y, unbiased estimator: {:.2f}\n'.format(y.var(ddof=1)))


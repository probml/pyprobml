#https://seaborn.pydata.org/generated/seaborn.pairplot.html
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)
g = sns.pairplot(iris, hue="species")
#plt.savefig(os.path.join('figures', 'iris-scatterplot.pdf'))
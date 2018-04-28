#https://seaborn.pydata.org/generated/seaborn.pairplot.html
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from utils.util import save_fig

iris = sns.load_dataset("iris")
#g = sns.pairplot(iris)
g = sns.pairplot(iris, hue="species")
save_fig('iris-scatterplot.pdf')
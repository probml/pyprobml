


import superimport

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
    

iris = load_iris()
X = iris.data 
y = iris.target

# Convert to pandas dataframe 
df_iris = pd.DataFrame(data=iris.data, 
                    columns=['sepal_length', 'sepal_width', 
                             'petal_length', 'petal_width'])
df_iris['species'] = pd.Series(iris.target_names[y], dtype='category')

#df_iris = df_iris[df_iris['species'] != 'virginica']


corr = df_iris.corr() 
mask = np.tri(*corr.shape).T 
plt.figure()
#sns.heatmap(corr.abs(), mask=mask, annot=True, cmap='viridis')
sns.heatmap(corr, mask=mask, annot=True, cmap='viridis')
plt.savefig('../figures/iris_corr_mat.pdf', dpi=300, bbox_inches='tight');
plt.show()

cov = df_iris.cov() 
plt.figure()
sns.heatmap(cov, annot=True, cmap='viridis')
plt.savefig('../figures/iris_cov_mat.pdf', dpi=300, bbox_inches='tight');
plt.show()
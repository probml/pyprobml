


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



corr = df_iris[df_iris['species'] != 'virginica'].corr() 
mask = np.tri(*corr.shape).T 
sns.heatmap(corr.abs(), mask=mask, annot=True, cmap='viridis')
plt.savefig('../figures/iris_corr_mat.pdf', dpi=300, bbox_inches='tight');
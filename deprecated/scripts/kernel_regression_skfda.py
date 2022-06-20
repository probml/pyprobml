import superimport

from skfda import preprocessing as pp
from skfda import FDataGrid
import math
np.random.seed(0)
N = 100
x = 10*(np.linspace(-1,1,100).reshape(-1,1))
ytrue = np.array([math.sin(abs(el))/(abs(el)) for el in x]).reshape(-1,1)
noise = 0.1
y	= ytrue + noise*np.random.randn(N,1)
x = (x - x.mean()) / x.std() # normalizing.

plt.figure(0)
plt.plot(x,ytrue)
plt.plot(x,y,'kx')

data_matrix = y.transpose()
grid_points = x.transpose()
fd = FDataGrid(data_matrix, grid_points) # smoother accepts this format of data.
nws = pp.smoothing.kernel_smoothers.NadarayaWatsonSmoother() # Smoother.
fd_estimate = nws.fit_transform(fd) # fitted datagrid, estimate will be in '<attr> data_matrix'

plt.plot(x,fd_estimate.data_matrix[0],'g--')
plt.legend(['true','data','estimate'])
plt.title('Gaussian kernel regression')
plt.savefig("/pyprobml/figures/kernelRegressionDemo.pdf",  dpi=300)

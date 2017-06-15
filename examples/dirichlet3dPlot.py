import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from scipy.stats import dirichlet
import os

grain = 250 #how many points along each axis to plot
edgedist = 0.008 #How close to an extreme value of say [1,0,0] are we willing to plot.
weight = np.linspace(0, 1, grain)

#Dirichlet parameter
alpha = 0.1
alphavec = np.array([1, 1, 1])*alpha

#Most extreme corners of the sample space
Corner1 = np.array([1.0 - edgedist*2, edgedist, edgedist])
Corner2 = np.array([edgedist, 1.0 - edgedist*2, edgedist])
Corner3 = np.array([edgedist, edgedist, 1.0 - edgedist*2])

#Probability density function that accepts 2D coordiantes
def dpdf(v1,v2):
    if (v1 + v2)>1:
        out = np.nan
    else:
        vec = v1 * Corner1 + v2 * Corner2 + (1.0 - v1 - v2)*Corner3
        out = dirichlet.pdf(vec, alphavec)
    return(out)

probs = np.array([dpdf(v1, v2) for v1 in weight for v2 in weight]).reshape(-1,grain)

fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111, projection='3d')
X,Y = np.meshgrid(weight, weight)
ax.plot_surface(Y, X, probs, cmap = 'jet', vmin=0, vmax=3,rstride=1,cstride=1, linewidth=0)
ax.view_init(elev=25, azim=230)
#ax.view_init(elev=25, azim=20)
ax.set_zlabel('p')
ax.set_title(r'$\alpha$'+'='+str(alpha))
plt.show()
plt.savefig(os.path.join('figures', 'DirSimplex%d.pdf' % (alpha*10)))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal

Gs = ["full", "diagonal", "spherical"]

# Gaussian Parameters:
# Mean:
mu = [0, 0]
# Covariances:
Covs = {'full': [[2, 1.8], [1.8, 2]],
        'diagonal': [[1, 0], [0, 3]],
        'spherical': [[1, 0], [0, 1]]}

#Multivariate gaussian PDF
def Gpdf(x, y, G):
    return multivariate_normal(mean=mu, cov=Covs[G]).pdf([x, y])

Gpdf = np.vectorize(Gpdf, excluded=['G'])

points = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(points, points)

def MakeGraph(G):
    Z = Gpdf(X, Y, G)
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z)
    plt.title(G)
    plt.draw()
    plt.savefig('Gaussian2D' + G)

for g in Gs:
    MakeGraph(g)

# Surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Z = Gpdf(X, Y, "spherical")

ax.plot_surface(X, Y, Z, rstride=2, cstride=2, color='white', edgecolor="black")
plt.savefig('Gaussian2DSurface')

plt.show()

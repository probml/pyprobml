import numpy as np
import scipy.spatial
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os

#This class comes from http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

x = [1, 0, 0]
y = [0, 1, 0]
z = [0, 0, 1]

pts = np.vstack([x,y]).T
tess = scipy.spatial.Delaunay(pts)

tri = tess.vertices
triang = mtri.Triangulation(x=pts[:, 0],y=pts[:,1], triangles=tri)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(triang, z, alpha = .3, color = 'red', edgecolors = 'blue')
ax.set_axis_off()

for i in range(3):
    EndPs = [[0,0],[0,0],[0,0]]
    EndPs[i][1] = 1.4
    art = Arrow3D(EndPs[0], EndPs[1], EndPs[2], mutation_scale=20, lw=3, arrowstyle="-|>", color="black")
    ax.add_artist(art)
    theta = '$\theta_' + str(i) + '$'
    EndPs = [[0,0],[0,0],[0,0]]
    if i == 0:
        EndPs[i][1] = 1
        EndPs[2][1] = -.2
    else:
        EndPs[i][1] = 1
    ax.text(EndPs[0][1], EndPs[1][1], EndPs[2][1], r'$\theta_%s$' % (i + 1),size=20)

ax.view_init(elev=30, azim=20)
ax.dist = 15
plt.draw()    

plt.savefig(os.path.join('figures', 'Simplex.pdf'))

#The code below comes from https://gist.github.com/tboggs/8778945
_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_triangle = mtri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0 for i in range(3)]

def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75 for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)

class Dirichlet(object):
    def __init__(self, alpha):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) /reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])

def draw_pdf_contours(dist, border=False, nlevels=200, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).
    Arguments:
        `dist`: A distribution instance with a `pdf` method.
        `border` (bool): If True, the simplex border is drawn.
        `nlevels` (int): Number of contours to draw.
        `subdiv` (int): Number of recursive mesh subdivisions to create.
        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    from matplotlib import ticker, cm

    refiner = mtri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.hold(1)
        plt.triplot(_triangle, linewidth=1)

def plot_points(X, barycentric=True, border=True, **kwargs):
    '''Plots a set of points in the simplex.
    Arguments:
        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.
        `barycentric` (bool): Indicates if `X` is in barycentric coords.
        `border` (bool): If True, the simplex border is drawn.
        kwargs: Keyword args passed on to `plt.plot`.
    '''
    if barycentric is True:
        X = X.dot(_corners)
    plt.plot(X[:, 0], X[:, 1], 'k.', ms=1, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.hold(1)
        plt.triplot(_triangle, linewidth=1)

def DrawDirAndSave(alpha):
    f = plt.figure()
    draw_pdf_contours(Dirichlet(alpha))
    plt.draw()
    plt.savefig(os.path.join('figures', 'dirichlet' + ''.join(str(e) for e in alpha) + '.pdf'))

#Plotting the dirichlet heatmaps.
DrawDirAndSave([2,2,2])
DrawDirAndSave([20,2,2])
plt.show()

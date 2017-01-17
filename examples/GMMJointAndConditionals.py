from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# 3 Component GMM Parameters:
ComponentNs = ['G1', 'G2', 'G3']
# Mixtures:
Pis = {'G1': .5,
       'G2': .3,
       'G3': .2}
# Means - first element refers to y, second refers to x.
Mus = {'G1': [0.22, 0.45],
       'G2': [0.5, 0.5],
       'G3': [0.77, 0.55]}
# Covariances:
Covs = {'G1': [[0.011, -0.01], [-0.01, 0.018]],
        'G2': [[0.018, 0.01], [0.01, 0.011]],
        'G3': [[0.011, -0.01], [-0.01, 0.018]]}

# Value that we will condition x on:
xcondval = .75

# Value that we will condition y on:
ycondval = .5

def GMMpdf(y, x):
    # For given y and x values, this returns the joint prob density
    density = 0
    for g in ComponentNs:
        denPart = multivariate_normal(mean=Mus[g], cov=Covs[g])
        density += Pis[g] * denPart.pdf([y, x])
    return density

def GMMConditional(which, val, condval):
    # If which == 0, this returns the prob y = val given x = condval
    # If which == 1, this returns the prob x = val given y = condval
    Other = (which + 1) % 2
    if which == 0:
        UnnormDensity = GMMpdf(val, condval)
    else:
        UnnormDensity = GMMpdf(condval, val)
    normalizer = 0
    for g in ComponentNs:
        normPart = multivariate_normal(mean=Mus[g][Other], cov=Covs[g][Other][Other])
        normalizer += Pis[g] * normPart.pdf(condval)
    density = UnnormDensity/normalizer
    return density

GMMpdf = np.vectorize(GMMpdf)
GMMConditional = np.vectorize(GMMConditional, excluded=['which', 'condval'])

spacing = np.arange(0, 1, 1.0/50)

Y, X = np.meshgrid(spacing, spacing)
Z = GMMpdf(Y, X)

# Gets rid of the background grid lines
mpl.rcParams['lines.linewidth'] = 0

# Forming joint graphic:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='white', edgecolor="black")
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
#Hiding some plot elements:
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.set_zticks([])
plt.draw()
plt.savefig('GMMJoint')

# Forming conditional graphic:

Cond1 = GMMConditional(0, spacing, xcondval)
Cond2 = GMMConditional(1, spacing, ycondval)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(spacing, Cond1, lw=2, color='black')
ax2.plot(spacing, Cond2, lw=2, color='black')
ax1.set_ylim([0, 5])
ax2.set_ylim([0, 5])

ax1.set_title('y given x = ' + str(xcondval), fontsize=18)
ax2.set_title('x given y = ' + str(ycondval), fontsize=18)
plt.draw()
plt.savefig('GMMConditional')

plt.show()

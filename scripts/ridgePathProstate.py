from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from scipy.io import loadmat
from sklearn import linear_model
import jax.numpy as jnp
import numpy as np

def get_weights(X, y, alphas):
    w = np.zeros((len(alphas),X.shape[1]))
    for i in range(len(alphas)):
        w[i,:] = Ridge(alpha=alphas[i]).fit(X, y).coef_
            return w


data = loadmat('../data/prostate/prostateStnd')
X, y, names = data['X'], data['y'], np.r_[[name[0] for name in (data['names'][0])]]

n_alphas, n_alphas_cv = 20, 30
alphas, alphas_cv = jnp.append(np.logspace(4, 0, n_alphas),jnp.zeros((1,1))), jnp.logspace(5, 0, n_alphas_cv)


w = get_weights(X, y, alphas)
variance = jnp.std(X, axis=0) **2
dof = np.c_[[jnp.sum(variance / (variance + alpha)) for alpha in alphas]]

plt.figure(1)
ax = plt.gca()
ax.plot(dof, w,'-o', linewidth=2)
ax.set_xscale('log')
plt.legend(names[:X.shape[1]])
plt.xlabel('dof', fontsize=18)
plt.ylabel('regression weights', fontsize=16)
plt.title('Ridge path on prostate data')
plt.axis('tight')




w_cv = get_weights(X, y, alphas_cv)
ridge_cv = linear_model.RidgeCV(alphas=alphas_cv, cv=5)
ridge_cv.fit(X, y)

plt.figure(2)
ax = plt.gca()
ax.plot( w_cv,'-o', linewidth=2)
plt.axvline(x=jnp.where(alphas_cv==ridge_cv.alpha_)[0][0], c="r", linewidth=2)
plt.legend(names[:X.shape[1]])
plt.axis('tight')
plt.show()

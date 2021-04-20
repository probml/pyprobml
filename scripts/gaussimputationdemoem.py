import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
from functools import reduce
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def is_pos_def(x):
     j = np.linalg.eigvals(x)
     return np.all(j>0)

def gauss_sample(mu, sigma, n):
     a = np.linalg.cholesky(sigma)
     z = np.random.randn(len(mu), n)
     k = np.dot(a, z)
     return np.transpose(mu + k)

def gauss_condition(mu, sigma, visible_nodes, visible_values):
     d = len(mu)
     j = np.array(range(d))
     v = visible_nodes.reshape(len(visible_nodes))
     h = np.setdiff1d(j, v)
     if len(h)==0:
         mugivh = np.array([])
         sigivh = np.array([])
     elif len(v) == 0:
         mugivh = mu
         sigivh = sigma
     else:
         ndx_hh = np.ix_(h, h)
         sigma_hh = sigma[ndx_hh]
         ndx_hv = np.ix_(h, v)
         sigma_hv = sigma[ndx_hv]
         ndx_vv = np.ix_(v, v)
         sigma_vv = sigma[ndx_vv]
         sigma_vv_inv = np.linalg.inv(sigma_vv)
         visible_values_len = len(visible_values)
         mugivh = mu[h] + np.dot(sigma_hv, np.dot(sigma_vv_inv, (visible_values.reshape((visible_values_len,1))-mu[v].reshape((visible_values_len, 1)))))
         sigivh = sigma_hh - np.dot(sigma_hv, np.dot(sigma_vv_inv, np.transpose(sigma_hv)))
     return mugivh, sigivh

def gauss_impute(mu, sigma, x):
     n_data, data_dim = x.shape
     x_imputed = np.copy(x)
     for i in range(n_data):
         hidden_nodes = np.argwhere(np.isnan(x[i, :]))
         visible_nodes = np.argwhere(~np.isnan(x[i, :]))
         visible_values = np.zeros(len(visible_nodes))
         for tc, h in enumerate(visible_nodes):
             visible_values[tc] = x[i, h]
         mu_hgv, sigma_hgv = gauss_condition(mu, sigma, visible_nodes, visible_values)
         for rr, h in enumerate(hidden_nodes):
             x_imputed[i, h] = mu_hgv[rr]
     return x_imputed

def impute_em(X, max_iter = 100, eps = 1e-04):

    nr, nc = X.shape
    C = np.isnan(X) == False                    # Identifying nan locations
    e = 0.0000001
    one_to_nc = np.arange(1, nc + 1, step = 1)
    M = one_to_nc * (C == False) - 1            # Missing locations (-1 at locations where Nan is present in X)
    O = one_to_nc * C - 1                       # Observed locations (-1 at locations where Nan is not present in X)

    # Generate Mu_0 and Sigma_0
    Mu = np.nanmean(X, axis = 0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows, ].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis = 0))
    
    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i, ]) != set(one_to_nc - 1): # Missing component exists
               
                m_indx = M[i, ] != -1
                o_indx = O[i, ] != -1 
                M_i = M[i, ][m_indx] # Missing entries
                O_i = O[i, ][o_indx] # Observed entries

                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)] + e # Ensuring invertibility

                Mu_tilde[i] = Mu[np.ix_(M_i)] + S_MO @ np.linalg.inv(S_OO) @ (X_tilde[i, O_i] - Mu[np.ix_(O_i)]) # Expected stats for mean
                X_tilde[i, M_i] = Mu_tilde[i]  # Storing mean in the missing entries
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM  
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O  
        
        Mu_new = np.mean(X_tilde, axis = 0)  # Updating mu 
        S_new = np.cov(X_tilde.T, bias = 1) + reduce(np.add, S_tilde.values()) / nr  # Updating sigma 
        no_conv = np.linalg.norm(Mu - Mu_new) >= eps or np.linalg.norm(S - S_new, ord = 2) >= eps # Convergence condition
        Mu = Mu_new
        S = S_new
        iteration += 1
    
    result = {'mu': Mu, 'Sigma': S}
    
    return result

np.random.seed(2) 
data_dim = 4
n_data = 100
threshold_missing = 0.5
mu = np.random.randn(data_dim, 1)
sigma = make_spd_matrix(n_dim=data_dim) 
x_full = gauss_sample(mu, sigma, n_data)
missing = np.random.rand(n_data, data_dim) < threshold_missing
x_miss = np.copy(x_full)
x_miss[missing] = np.nan
x_impute_oracle = gauss_impute(mu, sigma, x_miss)
result = impute_em(x_miss)
m = result.get('mu')
sig = result.get('Sigma')
m = m.reshape((-1, 1))
x_impute_em = gauss_impute(m, sig, x_miss)

"""##True params"""

r_squared = [] 
for i in range(4):
  miss = np.argwhere(np.isnan(x_miss[:, i]))
  r2 = r2_score(x_full[miss, i], x_impute_oracle[miss, i])
  r_squared.append(r2)

fig, axes = plt.subplots(nrows=2, ncols=2)
r=2
cnt = 0
reg = LinearRegression()
for i in range(r):
    miss = np.argwhere(np.isnan(x_miss[:, cnt]))
    min = x_full[miss, cnt].min()
    max = x_full[miss, cnt].max()
    xtest = np.linspace(min, max, 50).reshape(-1, 1)
    model = reg.fit(x_full[miss, cnt], x_impute_oracle[miss, cnt])
    line = model.predict(xtest)
    
    axes[i, 0].plot(xtest, line, color='black')
    axes[i, 0].scatter(x_full[miss, cnt], x_impute_oracle[miss, cnt], marker="*")
    axes[i, 0].set_title("R^2 = %5.3f" %(r_squared[cnt])) 
    axes[i, 0].set_xlabel("Truth")
    axes[i, 0].set_ylabel("Imputed")
    plt.tight_layout()

    cnt = cnt + 1

    miss = np.argwhere(np.isnan(x_miss[:, cnt]))
    min = x_full[miss, cnt].min()
    max = x_full[miss, cnt].max()
    xtest = np.linspace(min, max, 50).reshape(-1, 1)
    model = reg.fit(x_full[miss, cnt], x_impute_oracle[miss, cnt])
    line = model.predict(xtest)
    
    axes[i, 1].plot(xtest, line, color='black')
    axes[i, 1].scatter(x_full[miss, cnt], x_impute_oracle[miss, cnt], marker="*")
    axes[i, 1].set_title("R^2 = %5.3f" %(r_squared[cnt]))
    axes[i, 1].set_xlabel("Truth")
    axes[i, 1].set_ylabel("Imputed")
    plt.tight_layout()

    cnt = cnt + 1

fig.suptitle("Imputation with true parameters")
fig.tight_layout()
fig.subplots_adjust(top=0.85)
pml.save_fig('Imputation with true params.png')
plt.savefig('Imputation with true params.png')
plt.show()
"""##EM params"""

r_squared = [] 
for i in range(4):
  miss = np.argwhere(np.isnan(x_miss[:, i]))
  r2 = r2_score(x_full[miss, i], x_impute_em[miss, i])
  r_squared.append(r2)

fig, axes = plt.subplots(nrows=2, ncols=2)
plt.tight_layout()
r=2
cnt = 0
reg = LinearRegression()
for i in range(r):
    miss = np.argwhere(np.isnan(x_miss[:, cnt]))
    min = x_full[miss, cnt].min()
    max = x_full[miss, cnt].max()
    xtest = np.linspace(min, max, 50).reshape(-1, 1)
    model = reg.fit(x_full[miss, cnt], x_impute_em[miss, cnt])
    line = model.predict(xtest)
    
    axes[i, 0].plot(xtest, line, color='black')
    axes[i, 0].scatter(x_full[miss, cnt], x_impute_em[miss, cnt], marker="*")
    axes[i, 0].set_title("R^2 = %5.3f" %(r_squared[cnt])) #See this ok ....
    axes[i, 0].set_xlabel("Truth")
    axes[i, 0].set_ylabel("Imputed")
    plt.tight_layout()

    cnt = cnt + 1

    miss = np.argwhere(np.isnan(x_miss[:, cnt]))
    min = x_full[miss, cnt].min()
    max = x_full[miss, cnt].max()
    xtest = np.linspace(min, max, 50).reshape(-1, 1)
    model = reg.fit(x_full[miss, cnt], x_impute_em[miss, cnt])
    line = model.predict(xtest)
    
    axes[i, 1].plot(xtest, line, color='black')
    axes[i, 1].scatter(x_full[miss, cnt], x_impute_em[miss, cnt], marker="*")
    axes[i, 1].set_title("R^2 = %5.3f" %(r_squared[cnt]))
    axes[i, 1].set_xlabel("Truth")
    axes[i, 1].set_ylabel("Imputed")
    plt.tight_layout()

    cnt = cnt + 1

fig.suptitle("Imputation with EM parameters")
fig.tight_layout()
fig.subplots_adjust(top=0.85)
pml.save_fig('Imputation with EM params.png')
plt.savefig('Imputation with EM params.png')
plt.show()

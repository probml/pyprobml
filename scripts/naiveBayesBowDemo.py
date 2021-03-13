from pymatreader import read_mat
import scipy
import random
import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


data = None
Xtrain = None
Xtest = None

if True:
    data = read_mat('data\XwindowsDocData.mat')
    Xtrain = data['xtrain']
    Xtrain = scipy.sparse.csc_matrix.toarray(Xtrain)
    Xtest = data['xtest']
    Xtest = scipy.sparse.csc_matrix.toarray(Xtest)

'''
else:
    data = read_mat('data\20news_w100.mat')
    keys =  list(data.keys())
    X = random.shuffle(keys)
    Xtrain = X[1:60, :]; 
    Xtest = X[61:, :]; 
'''
ytrain = data['ytrain']
ytest = data['ytest']
model = GaussianNB()
ypred_train = model.fit(Xtrain, ytrain).predict(Xtrain)
err_train = np.mean(zero_one_loss(ytrain, ypred_train))
ypred_test = model.fit(Xtrain, ytrain).predict(Xtest)
err_test = np.mean(zero_one_loss(ytest, ypred_test))
print('misclassification rates  on train = '+str(err_train*100) +
      ' pc, on test = '+str(err_test*100)+' pc\n')


C = np.unique(data['ytrain']).size
for i in range(0, C):
    plt.bar(np.arange(0, 600, 1), model.theta_[i, :])
    plt.title('p(xj=1|y='+str(i)+')')
    # plt.savefig(r'..\figures\naiveBayesBow'+str(i)+'ClassCond')
    plt.show()

N = 5
Mp = np.zeros((N, C))
for c in range(0, C):
    sortedProb = (model.theta_[c, :])
    sortedProb.sort(axis=0)
    sortedProb = sortedProb[::-1]
    print('top '+str(N)+' words for class '+str(c+1))
    for w in range(0, N):
        # print(2+'+str(w)+' '+str(sortedProb[w])+' '+ +'\n', w,  sortedProb(w), vocab{ndx(w)})
        Mp[w, c] = sortedProb[w]
        # Mw[w,c] = vocab{ndx(w)}


mi = mutual_info_classif(Xtrain, ytrain)
mi.sort(axis=0)
mi = mi[::-1]
#sortedMI, ndx = mi
sortedMI = mi
print('top '+str(N)+' words sorted by MI')
Mi = np.zeros((N,))
for w in (0, N):
    # fprintf(2,'%2d %6.4f %20s\n', w,  sortedMI(w), vocab{ndx(w)});
    Mi[w] = sortedMI[w]
    # Miw{w} = vocab{ndx(w)};

M = np.zeros((N, 6))
for i in range(0, N):
    # M[i,1] = Mw[i,1]
    M[i, 2] = Mp[i, 1]
    # M{i,3} = Mw[i,2]
    M[i, 4] = Mp[i, 2]
    # M[i,5] = Miw[i]
    M[i, 6] = Mi[i]

'''
MM = [Mw(:, 1), mat2cellRows(Mp(:, 1)), Mw(:,2),  mat2cellRows(Mp(:, 2)), Miw(:),  mat2cellRows(Mi(:))];
assert(isequal(M, MM))

colLabels = {'class 1', 'prob', 'class 2', 'prob', 'highest MI', 'MI'};
latextable(M, 'horiz', colLabels, 'Hline',[1], 'format', '%5.3f');
'''

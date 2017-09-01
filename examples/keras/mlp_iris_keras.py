
# Fit an MLP  on 3 class Iris data


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

np.random.seed(123) # try to enforce reproduacability

# import the data 
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y_int = iris.target
nclasses = len(np.unique(Y_int))
Y = keras.utils.to_categorical(Y_int, num_classes=nclasses) # one hot

ncases = X.shape[0]
ndim = X.shape[1]

# MLP with 0 or 1 hidden layers
def mk_mlp(ninputs = ndim, nhidden = 100, learn_rate=0.01):
    model = Sequential()
    if nhidden > 0:
        model.add(Dense(nhidden, activation='relu', input_dim=ninputs))
        model.add(Dense(nclasses, activation='softmax'))
    else: # no hidden layer
        model.add(Dense(nclasses, activation='softmax', input_dim=ninputs))
    #opt = keras.optimizers.Adam(lr=learn_rate)
    opt = keras.optimizers.Nadam(lr=learn_rate)
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=opt,
            metrics=['accuracy'])
    return model


batch_sizes = [round(0.5*ncases),  ncases]
lrs = [0.005, 0.01]
nhiddens = [0, 10, 100]
for batch_size in batch_sizes:
    for lr in lrs:
        for nhidden in nhiddens:
            model = mk_mlp(nhidden = nhidden, learn_rate = lr)
            history = model.fit(X, Y, epochs=200, batch_size=batch_size, verbose=0)
            final_acc = history.history['acc'][-1]
            print(' batch {:0.3f}, lr {:0.3f}, nh {:0.3f}. Acc {:0.3f}'.format(
                batch_size, lr, nhidden, final_acc))
            #score = model.evaluate(X, Y, batch_size=ncases)
            #acc = score[1]
            #print('final  acc on trainset {0:.2f}'.format(acc))


#predicted = model.predict(X) # 150x3
#predicted = model.predict_classes(X) # 150x3
#accuracy_mlp = metrics.accuracy_score(Y_int, predicted)

# Plot loss over time
if False:
    loss_trace = history.history['loss']
    acc_trace = history.history['acc']
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(loss_trace)
    plt.title('loss')
    plt.subplot(1,2,2)
    plt.plot(acc_trace)
    plt.title('accuracy')
    plt.show()

# Now let us wrap it in scikit learn so we can do grid search
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
if False:
    model = KerasClassifier(build_fn=mk_mlp, epochs=100, verbose=0)
    batch_sizes = [batch_size] #[round(0.1*ncases), round(0.5*ncases)]
    lrs = [0.001, 0.01]
    param_grid = dict(batch_size=batch_sizes, learn_rate=lrs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
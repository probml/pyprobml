#https://medium.com/@ilblackdragon/tensorflow-tutorial-part-3-c5fc0662bc08#.r00rs5mv9
import random
import pandas
import numpy as np
import tensorflow as tf
from sklearn import metrics, cross_validation

#import skflow
import tensorflow.contrib.learn as skflow

random.seed(42)

#data = pandas.read_csv('titanic_train.csv')
data = pandas.read_csv('../data/titanic_train.csv')
X = data[["Embarked"]]
y = data["Survived"]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)

embarked_classes = X_train["Embarked"].unique()
print('Embarked has next classes: ', embarked_classes)

cat_processor = skflow.preprocessing.CategoricalProcessor()
cp = cat_processor.fit_transform(X_train)

X_train = np.array(list(cat_processor.fit_transform(X_train)))
X_test = np.array(list(cat_processor.transform(X_test)))

# Total number of classes for this variable from Categorical Processor.
# Includes unknown token and unique classes for variable.
n_classes = len(cat_processor.vocabularies_[0])

### Embeddings

EMBEDDING_SIZE = 3

def categorical_model(X, y):
    features = skflow.ops.categorical_variable(
        X, n_classes, embedding_size=EMBEDDING_SIZE, name='embarked')
    return skflow.models.logistic_regression(tf.squeeze(features, [1]), y)
# features has shape (712, 1, 3)

classifier = skflow.TensorFlowEstimator(model_fn=categorical_model,
    n_classes=2)
classifier.fit(X_train, y_train)

print("Accuracy: {0}".format(metrics.accuracy_score(classifier.predict(X_test), y_test)))
print("ROC: {0}".format(metrics.roc_auc_score(classifier.predict(X_test), y_test)))

### One Hot

def one_hot_categorical_model(X, y):
    features = skflow.ops.one_hot_matrix(X, n_classes)
    return skflow.models.logistic_regression(tf.squeeze(features, [1]), y)

classifier = skflow.TensorFlowEstimator(model_fn=one_hot_categorical_model,
    n_classes=2, steps=1000, learning_rate=0.01)
classifier.fit(X_train, y_train)

print("Accuracy: {0}".format(metrics.accuracy_score(classifier.predict(X_test), y_test)))
print("ROC: {0}".format(metrics.roc_auc_score(classifier.predict(X_test), y_test)))
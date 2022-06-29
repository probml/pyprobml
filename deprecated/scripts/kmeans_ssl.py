

# K-means clustering for semisupervised learning
# Code is from chapter 9 of 
# https://github.com/ageron/handson-ml2

import superimport

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# 1,797 grayscale 8×8 images representing digits 0 to 9.
X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

# Logreg on the pixels
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
score_baseline = log_reg.score(X_test, y_test)
print(score_baseline)


# Cluster the points, then use Logreg on distance to each cluster
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)
score_kmeans = pipeline.score(X_test, y_test)
              
print(score_kmeans)


# Logreg on the pixels, small labeled training set
n_labeled = 50
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
score_50 = log_reg.score(X_test, y_test)
print(score_50)

# Cluster the unalabled training set
k=50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

# Show the clusters
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')
#plt.savefig("representative_images_diagram", tight_layout=False)
plt.show()

# Manually label them
y_representative_digits = np.array([
   0,1,3,2,7,6,4,6,9,5,
   1,2,9,5,2,7,8,1,8,6,
   3,1,5,4,5,4,0,3,2,6,
   1,7,7,9,1,8,6,5,4,8,
   5,3,3,6,7,9,7,8,4,9])

# Train on the represenative images
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
score_manual = log_reg.score(X_test, y_test)
print(score_manual)


y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
    
# Propaget label to all points in top 20% of prxomity to  cluster center
percentile_closest = 100
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
score_prop = log_reg.score(X_test, y_test)
print(score_prop)

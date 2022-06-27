# Relevance network newsgroup
# Author: Drishtii@
# Based on:
# https://github.com/probml/pmtk3/blob/master/demos/relevanceNetworkNewsgroupDemo.m

#!pip install pgmpy

import superimport

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import networkx as nx
from pgmpy.estimators import TreeSearch
from sklearn.feature_extraction.text import CountVectorizer
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from IPython.display import Image, display
import pyprobml_utils as pml
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score

newsgroups_train = fetch_20newsgroups(subset='train')

list_of_words = ['baseball', 'bible', 'case','course','evidence','children','mission','launch','files',
'games','league', 'nhl','fans', 'hockey','players','christian','fact','god', 'human', 'jews', 
'war', 'president', 'law', 'orbit', 'shuttle', 'moon', 'program', 'version', 'graphics', 'video',
'israel','government','earth','gun', 'nasa','lunar','format', 'ftp', 'card','jesus','computer', 'science',
'religion', 'world', 'rights', 'solar', 'space', 'windows', 'state']

count_vect = CountVectorizer(newsgroups_train.data, vocabulary=list_of_words)   
X_train_counts = count_vect.fit_transform(newsgroups_train.data)

df_ = pd.DataFrame.sparse.from_spmatrix(X_train_counts, columns=list_of_words)

n_jobs = 1
edge_weights_fn = mutual_info_score
data = df_
pbar = combinations(df_.columns, 2)
n_vars = len(df_.columns)

vals = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(edge_weights_fn)(data.loc[:, u], data.loc[:, v]) for u, v in pbar)
weights = np.zeros((n_vars, n_vars))
weights[np.triu_indices(n_vars, k=1)] = vals
max = np.max(weights)
twenty_percent_of_max = 0.2*max 

# Considering edges whose mutual information is greater than or equal to 20% of the maximum pairwise MI
final_weights = np.zeros((n_vars, n_vars))
for i in range(n_vars):
  for j in range(n_vars):
    if (weights[i, j] > twenty_percent_of_max):
      final_weights[i, j] = weights[i, j]

G = nx.from_numpy_array(final_weights, create_using=nx.MultiGraph)
G.remove_nodes_from(list(nx.isolates(G)))

keys = list(G.nodes)
values = list_of_words
dictionary = dict(zip(keys, values))
G = nx.relabel_nodes(G, dictionary)

def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)

p2=nx.drawing.nx_pydot.to_pydot(G)
view_pydot(p2)
p2.write_png('../figures/relevance_network.png')

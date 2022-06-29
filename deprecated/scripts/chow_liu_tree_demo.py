# Chow-Liu algorithm 
# Author: Drishtii@
# Based on
# https://github.com/probml/pmtk3/blob/master/demos/chowliuTreeDemo.m

#!pip install pgmpy


import superimport

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import networkx as nx
from pgmpy.estimators import TreeSearch
from sklearn.feature_extraction.text import CountVectorizer
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from IPython.display import Image, display
import pyprobml_utils as pml

newsgroups_train = fetch_20newsgroups(subset='train')

list_of_words = ['email','disk','ftp','files','format','image','display','phone','number','card', 'graphics','windows','driver', 'pc','drive', 'memory','scsi', 
'data','system','problem','dos','program','space','version','win','team', 'won','car','video', 'software','bmw', 'dealer', 'engine', 'honda', 'mac', 'help', 'server',
'launch', 'moon', 'nasa', 'orbit', 'shuttle', 'technology', 'fans', 'games', 'hockey', 'league', 'players', 'puck', 'season','oil','lunar','bible','children',
'mars', 'earth','god','satellite', 'solar','mission', 'nhl','war','world','science','computer','baseball','hit','christian','power','jesus', 'religion','jews',
'government','israel','state','university','research','question','aids','msg','food','water','health','insurance','patients','medicine','studies','case','president',
'human','fact','course','rights','law','gun','evidence']

count_vect = CountVectorizer(newsgroups_train.data, vocabulary=list_of_words)   
X_train_counts = count_vect.fit_transform(newsgroups_train.data)
df_ = pd.DataFrame.sparse.from_spmatrix(X_train_counts, columns=list_of_words)

# Learning graph structure 
est = TreeSearch(df_, root_node='email')
dag = est.estimate(estimator_type="chow-liu")

# Plot and display
def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)

p=nx.drawing.nx_pydot.to_pydot(dag)
view_pydot(p)
p.write_png('../figures/tree_structure.png')
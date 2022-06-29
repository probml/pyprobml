# Fit a classifier to the 20 newsgroup dataset

#https://scikit-learn.org/dev/tutorial/text_analytics/working_with_text_data.html

#
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html

import superimport

import numpy as np

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)

print("Num. training docs {}".format(len(data.data))) # 2257


for line in data.data[0].split("\n"): print(line)

"""
Doc 0 has length 21, class id is 1, label is comp.graphics

From: sd345@city.ac.uk (Michael Collier)
Subject: Converting images to HP LaserJet III?
Nntp-Posting-Host: hampton
Organization: The City University
Lines: 14

Does anyone know of a good way (standard PC application/PD utility) to
convert tif/img/tga files into LaserJet III format.  We would also like to
do the same, converting to HPGL (HP plotter) files.

Please email any response.

Is this the correct group?

Thanks in advance.  Michael.
-- 
Michael Collier (Programmer)                 The Computer Unit,
Email: M.P.Collier@uk.ac.city                The City University,
Tel: 071 477-8000 x3769                      London,
Fax: 071 477-8565                            EC1V 0HB.
"""



#from sklearn.model_selection import train_test_split
X_train = data.data
y_train = data.target

data_test = fetch_20newsgroups(subset='test',
     categories=categories, shuffle=True, random_state=42)
X_test = data_test.data
y_test = data_test.target

print("Num. testing docs {}".format(len(data_test.data))) # 1502

###################
# Fit logreg using tfidf

from sklearn.pipeline import Pipeline
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
pipeline = Pipeline([
    ('bow', CountVectorizer(tokenizer=nltk.word_tokenize)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True))
    ])
X_train_tfidf = pipeline.fit_transform(X_train) 
print(np.shape(X_train_tfidf)) # (2257, 42100)
X_test_tfidf = pipeline.transform(X_test)
print(np.shape(X_test_tfidf)) # (1502, 42100)

from sklearn.linear_model import LogisticRegression
# For simplicity, we set the L2 regularizer to a constant lambda=1/C=1e-3
logreg_tfidf = LogisticRegression(C=1e3, solver='lbfgs', multi_class='multinomial')
logreg_tfidf.fit(X_train_tfidf, y_train)
ypred_tfidf = logreg_tfidf.predict(X_test_tfidf)
accuracy_tfidf =  np.mean(ypred_tfidf == y_test) 
print(accuracy_tfidf) # 90.6%
# According to https://scikit-learn.org/dev/tutorial/text_analytics/working_with_text_data.html,
# multinomial naive Bayes (on tf-idf rep) gets  83.5% and an SVM gets 91% 

###############
# Now use word embeddings
# https://spacy.io/

import spacy
nlp = spacy.load('en_core_web_md', disable=['tagger','parser','ner']) # Just tokenize
X_train_embed = [nlp(doc).vector for doc in X_train]
print(np.shape(X_train_embed)) # (2257, 300)
X_test_embed = [nlp(doc).vector for doc in X_test]
print(np.shape(X_test_embed)) # (1502, 300)

from sklearn.linear_model import LogisticRegression
# For simplicity, we set the L2 regularizer to a constant lambda=1/C=1e-3
logreg_embed = LogisticRegression(C=1e3, solver='lbfgs', multi_class='multinomial')
logreg_embed.fit(X_train_embed, y_train)
ypred_embed = logreg_embed.predict(X_test_embed)
accuracy_embed =  np.mean(ypred_embed == y_test) 
print(accuracy_embed) # 86.9%



import numpy as np

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# http://www.pitt.edu/~naraehan/presentation/Movie+Reviews+sentiment+analysis+with+Scikit-Learn.html
# https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d

corpus = [
   'This is the first document, I say.',
   'This document is the second document',
   'And this is the 3 one!',
   'Is this the first document, you ask?',
   ]


from sklearn.feature_extraction.text import CountVectorizer

# default tokenzier drops "I" as well as punctuation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
# ['and', 'ask', 'document', 'first', 'is', 'one', 'say', 'second', 'the', 'this', 'you']


# NLTK tokenzier keeps everything
import nltk
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
#vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, min_df=1, max_features = 5)
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
#['!', ',', '.', '3', '?', 'and', 'ask', 'document', 'first', 'i', 'is', 'one', 'say', 'second', 'the', 'this', 'you']

print(X.todense())
"""
[[0 1 1 0 0 0 0 1 1 1 1 0 1 0 1 1 0]
 [0 0 0 0 0 0 0 2 0 0 1 0 0 1 1 1 0]
 [1 0 0 1 0 1 0 0 0 0 1 1 0 0 1 1 0]
 [0 1 0 0 1 0 1 1 1 0 1 0 0 0 1 1 1]]
"""

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X)
Xtf = tf_transformer.transform(X) 
# Xtf[i,j] = L2-normalize(tf[i,:])_j 
tf = X.todense()
from sklearn.preprocessing import normalize
Xpred1 = normalize(tf)
Xpred2 = tf / np.sqrt(np.sum(tf**2,axis=1))
assert np.isclose(Xtf.todense(), Xpred1).all()
assert np.isclose(Xtf.todense(), Xpred2).all()
print(np.round(Xtf.todense(),3))
"""
[[0.    0.333 0.333 0.    0.    0.    0.    0.333 0.333 0.333 0.333 0.
  0.333 0.    0.333 0.333 0.   ]
 [0.    0.    0.    0.    0.    0.    0.    0.707 0.    0.    0.354 0.
  0.    0.354 0.354 0.354 0.   ]
 [0.378 0.    0.    0.378 0.    0.378 0.    0.    0.    0.    0.378 0.378
  0.    0.    0.378 0.378 0.   ]
 [0.    0.333 0.    0.    0.333 0.    0.333 0.333 0.333 0.    0.333 0.
  0.    0.    0.333 0.333 0.333]]
"""


tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
Xtfidf = tfidf_transformer.fit_transform(X) 
Xbin = (X>0).todense() # XX[i,j]=1 iff word j occurs at least once in doc i
df = np.sum(Xbin, axis=0)
n = np.shape(X)[0]
idf = np.log( (1+n) / (1+df) ) + 1
assert np.isclose(idf, tfidf_transformer.idf_).all()

tf = X.todense()
tmp= np.multiply(tf, idf) # Xcounts is (N,D), idf is (1,D)
B = tf * tfidf_transformer._idf_diag
assert np.isclose(tmp, B).all()
tfidf = normalize(tmp)
assert np.isclose(tfidf, Xtfidf.todense()).all()


# Make a pipeline

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(tokenizer=nltk.word_tokenize)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True))
    ])
Xtrain = pipeline.fit_transform(corpus)
assert np.isclose(Xtfidf.todense(), Xtrain.todense()).all()


corpus_test = [
   'This document is the first new document.',
   'And this is the second.'
   ]
Xtest = pipeline.transform(corpus_test)
print(np.round(Xtest.todense(),3))
"""
[[0.    0.    0.496 0.    0.    0.    0.    0.633 0.391 0.    0.259 0.
  0.    0.    0.259 0.259 0.   ]
 [0.    0.    0.512 0.    0.    0.512 0.    0.    0.    0.    0.267 0.
  0.    0.512 0.267 0.267 0.   ]]
"""

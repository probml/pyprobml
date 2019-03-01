
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# http://www.pitt.edu/~naraehan/presentation/Movie+Reviews+sentiment+analysis+with+Scikit-Learn.html
# https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d

import numpy as np
from numpy.testing import assert_allclose

corpus = [
   'This is the first example.',
   'This example is the second example',
   'Do you want to see more examples, or is three examples enough?',
   ]


from sklearn.feature_extraction.text import CountVectorizer
# default tokenizer drops important words, NLTK tokenzier keeps everything, 
from nltk.tokenize import RegexpTokenizer
tokenizer = lambda s: RegexpTokenizer(r'\w+').tokenize(s) # alphanumeric strings get tokenized
vectorizer = CountVectorizer(tokenizer=tokenizer)
B = vectorizer.fit_transform(corpus).todense() # bag of words, (N,T)
print(vectorizer.get_feature_names())
['do', 'enough', 'example', 'examples', 'first', 'is', 'more', 'or', 'second', 'see', 'the', 'this', 'three', 'to', 'want', 'you']

print(B)
"""
[[0 0 1 0 1 1 0 0 0 0 1 1 0 0 0 0]
 [0 0 2 0 0 1 0 0 1 0 1 1 0 0 0 0]
 [1 1 0 2 0 1 1 1 0 1 0 0 1 1 1 1]]
"""

## TF transform
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(B)
Btf = tf_transformer.transform(B).todense()
# Compute TF matrix "manually"
# Btf[i,j] = L2-normalize(tf[i,:])_j 
from sklearn.preprocessing import normalize
assert_allclose(Btf, normalize(B), atol=1e-2)
assert_allclose(Btf, B / np.sqrt(np.sum(np.power(B,2),axis=1)), atol=1e-2)


## TF-IDF transform
tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
Btfidf = tfidf_transformer.fit_transform(B).todense()
# Compute idf "manually"
Bbin = (B>0) # Bbin[i,j]=1 iff word j occurs at least once in doc i
df = np.ravel(np.sum(Bbin, axis=0)) # convert from (1,T) to (T)
n = np.shape(B)[0]
idf = np.log( (1+n) / (1+df) ) + 1
assert_allclose(idf, tfidf_transformer.idf_, atol=1e-2)
# Compute tf-idf "manually"
tfidf = normalize(np.multiply(B, idf))
assert_allclose(tfidf, Btfidf, atol=1e-2)


# Make a pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(tokenizer=tokenizer)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True))
    ])
Btrain = pipeline.fit_transform(corpus).todense()
assert_allclose(Btfidf, Btrain)

corpus_test = [
   'This example is a new document.',
   'And this is the second test.'
   ]
Btest = pipeline.transform(corpus_test)
print(np.round(Btest.todense(),3))
"""
[[0.    0.    0.62  0.    0.    0.481 0.    0.    0.    0.    0.    0.62
  0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.373 0.    0.    0.632 0.    0.48  0.48
  0.    0.    0.    0.   ]]
"""

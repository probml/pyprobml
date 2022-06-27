
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# http://www.pitt.edu/~naraehan/presentation/Movie+Reviews+sentiment+analysis+with+Scikit-Learn.html
# https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d

import superimport

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

from tensorflow import keras
t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(corpus)
print(t.document_count)
print(t.word_counts)
print(t.word_docs)
print(t.word_index)
"""
3
OrderedDict([('this', 2), ('is', 3), ('the', 2), ('first', 1), ('example', 3),
 ('second', 1), ('do', 1), ('you', 1), ('want', 1), ('to', 1),
 ('see', 1), ('more', 1), ('examples', 2), ('or', 1), ('three', 1), ('enough', 1)])
defaultdict(<class 'int'>, {'first': 1, 'the': 2, 'is': 3, 'this': 2, 'example': 2, 
'second': 1, 'you': 1, 'see': 1, 'do': 1, 'or': 1, 'examples': 1, 'enough': 1,
 'three': 1, 'more': 1, 'want': 1, 'to': 1})
{'is': 1, 'example': 2, 'this': 3, 'the': 4, 'examples': 5, 'first': 6, 
'second': 7, 'do': 8, 'you': 9, 'want': 10, 'to': 11, 'see': 12, 'more': 13,
 'or': 14, 'three': 15, 'enough': 16}
"""

encoded_docs = t.texts_to_matrix(corpus, mode='count')
print(encoded_docs)
"""
[[0. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 2. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
"""
reverse_word_index = dict([(value, key) for (key, value) in t.word_index.items()])
"""
{1: 'is',
 2: 'example',
 3: 'this',
 4: 'the',
 5: 'examples',
 6: 'first',
 7: 'second',
 8: 'do',
 9: 'you',
 10: 'want',
 11: 'to',
 12: 'see',
 13: 'more',
 14: 'or',
 15: 'three',
 16: 'enough'}
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

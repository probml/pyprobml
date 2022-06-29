
# Demo of word embeddigns using the Spacy library 
# Based on https://spacy.io/usage/vectors-similarity
# and https://nlpforhackers.io/complete-guide-to-spacy/

# Follow installation instructions at https://spacy.io/usage/
# Then run the command below to get a word embedding model (medium sized)
# python -m spacy download en_core_web_md

import superimport

import spacy
import numpy as np
import pandas as pd
from scipy import spatial
 
nlp = spacy.load('en_core_web_md', disable=['tagger','parser','ner']) # Just tokenize

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

# Pairwise similarity
tokens = nlp(u'dog cat banana') # type spacy.tokens.doc.Doc
N = len(tokens)
S = np.empty((N,N))
S2 = np.empty((N,N))
for i in range(N):
  for j in range(N):
    t1 = tokens[i]
    t2 = tokens[j]
    S[i,j] = t1.similarity(t2)
    S2[i,j] = cosine_similarity(t1.vector, t2.vector)

assert np.isclose(S, S2).all()
df=pd.DataFrame(data=S,columns=tokens,index=tokens)
print(df)
"""
             dog       cat    banana
dog     1.000000  0.801686  0.243276
cat     0.801686  1.000000  0.281544
banana  0.243276  0.281544  1.000000
"""



# The vector embedding of a doc is the average of the vectors of each token
# Each token has a D=300 dimensional embedding
tokens = nlp(u"The cat sat on the mat.")
token_embeds = np.array([t.vector for t in tokens]) # N*D
assert np.isclose(tokens.vector, np.mean(token_embeds,axis=0)).all()

# we can use this for document retrieval

target = nlp("Cats are beautiful animals.")
doc1 = nlp("Dogs are awesome.")
doc2 = nlp("Some gorgeous creatures are felines.")
doc3 = nlp("Dolphins are swimming mammals.") 
print(target.similarity(doc1))  # 0.8901765218466683
print(target.similarity(doc2))  # 0.9115828449161616
print(target.similarity(doc3))  # 0.7822956752876101






# Vector space arithmetic


man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector
queen = nlp.vocab['queen'].vector
king = nlp.vocab['king'].vector

# We now need to find the closest vector in the vocabulary to the result of "man" - "woman" + "queen"
maybe_king = man - woman + queen
computed_similarities = []
from time import time
time_start = time() #  slow to search through 1,344,233 words
print("searching through {} words for nearest neighbors".format(len(nlp.vocab)))
for word in nlp.vocab:
    if not word.has_vector:
        continue
    similarity = cosine_similarity(maybe_king, word.vector)
    computed_similarities.append((word, similarity))
print('time spent training {:0.3f}'.format(time() - time_start))
 
computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
print([w[0].text for w in computed_similarities[:10]])
 
# ['Queen', 'QUEEN', 'queen', 'King', 'KING', 'king', 'KIng', 'KINGS', 'kings', 'Kings']

"""
X = [word.vector for word in nlp.vocab if word.has_vector]
print(np.shape(X)) # (684 755, 300)
N = 10000; D = 300;
X = np.random.rand(N,D)
# datascience book p89 - correct, but gets memory error if N is too large
time_start = time() 
sqdst = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis=-1)
print(time() - time_start)  
i=10;j=200;assert np.isclose(sqdst[i,j], np.sum((X[i,:]-X[j,:])**2))
"""



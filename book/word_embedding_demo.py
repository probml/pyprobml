# Demo of word embeddigns using the Spacy library 
# Based on https://spacy.io/usage/vectors-similarity

# Follow installation instructions at https://spacy.io/usage/
# Then run the command below to get a word embedding model
# python -m spacy download en_core_web_md

import spacy
import numpy as np
import pandas as pd

#nlp = spacy.load('en_core_web_md')  # the default small model does not have word vectors
nlp = spacy.load('en_core_web_md', disable=['tagger','parser','ner']) # Just tokenize

# The vector embedding of a doc is the average of the vectors of each token
# Each token has a D=300 dimensional embedding
tokens = nlp(u"I ain't in a hurry to go to Gdansk!")
token_embeds = np.array([t.vector for t in tokens]) # N*D
assert np.isclose(tokens.vector, np.mean(token_embeds,axis=0)).all()


# Pairwise similarity
tokens = nlp(u'dog cat banana') # type spacy.tokens.doc.Doc
N = len(tokens)
S = np.empty((N,N))
for i in range(N):
  for j in range(N):
    t1 = tokens[i]
    t2 = tokens[j]
    S[i,j] = t1.similarity(t2)
df=pd.DataFrame(data=S,columns=tokens,index=tokens)
print(df)
"""
             dog       cat    banana
dog     1.000000  0.801686  0.243276
cat     0.801686  1.000000  0.281544
banana  0.243276  0.281544  1.000000
"""


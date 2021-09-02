# Demo of the Spacy NLP library 
# Based on https://spacy.io/
# See also
# https://nlpforhackers.io/complete-guide-to-spacy/


import superimport

import spacy

nlps = spacy.load('en_core_web_sm')
nlpm = spacy.load('en_core_web_md')
tokens = nlpm(u'dog cat banana afskfsd')
for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
  
doc = nlps(u'Apple is looking at buying the U.K. startup FooCon for $1 billion.')  
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
"""
Apple apple PROPN NNP nsubj Xxxxx True False
is be VERB VBZ aux xx True True
looking look VERB VBG ROOT xxxx True False
at at ADP IN prep xx True True
buying buy VERB VBG pcomp xxxx True False
the the DET DT det xxx True True
U.K. u.k. PROPN NNP compound X.X. False False
startup startup NOUN NN dobj xxxx True False
FooCon foocon NOUN NN appos XxxXxx True False
for for ADP IN prep xxx True True
$ $ SYM $ quantmod $ False False
1 1 NUM CD compound d False False
billion billion NUM CD pobj xxxx True False
. . PUNCT . punct . False False
"""
    
# With the medium model, 'is' and 'at' are not flagged as stop words.
# This is a know bug.
# https://github.com/explosion/spaCy/issues/922
# Here is a fix.
nlpm.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
doc = nlpm(u'Apple is looking at buying the U.K. startup FooCon for $1 billion.')  
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)



corpus=[    
    'Mary had a little lamb, little lamb, little lamb',
    'Mary had a little lamb',
    'Whose fleece was white as snow.',
    'And everywhere that Mary went',
    'Mary went, Mary went,',
    'Everywhere that Mary went',
    'The lamb was sure to go.'
    ]
corpus_tokenized = [nlpm(doc) for doc in corpus]
all_tokens =  [token for doc in corpus_tokenized for token in doc]
#vocab = set(all_tokens)
vocab = set()
for t in all_tokens:
    vocab.add(str(t))


    
    
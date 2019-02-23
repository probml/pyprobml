#https://scikit-learn.org/dev/tutorial/text_analytics/working_with_text_data.html

import numpy as np

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)

X = data.data; # list of strings
Y = data.target; # numpy array of ints
print("Num. docs {}".format(len(X))) # 2257

def print_data(data, idx, nlines=20):
  X = data.data; # list of strings
  Y = data.target; # numpy array of ints
  x = X[idx] # a string
  y = Y[idx]
  label = data.target_names[y]
  lines = x.split("\n")
  print("Doc {} has length {}, class id is {}, label is {}".format(idx, len(lines), y, label))
  if not(nlines is None):
    lines = lines[:nlines]
  print("\n".join(lines))

  
print_data(data, 0)
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



print_data(data, 2)
"""
Doc 2 has length 69, class id is 3, label is soc.religion.christian

From: djohnson@cs.ucsd.edu (Darin Johnson)
Subject: Re: harrassed at work, could use some prayers
Organization: =CSE Dept., U.C. San Diego
Lines: 63

(Well, I'll email also, but this may apply to other people, so
I'll post also.)

>I've been working at this company for eight years in various
>engineering jobs.  I'm female.  Yesterday I counted and realized that
>on seven different occasions I've been sexually harrassed at this
>company.

>I dreaded coming back to work today.  What if my boss comes in to ask
>me some kind of question...

Your boss should be the person bring these problems to.  If he/she
does not seem to take any action, keep going up higher and higher.
Sexual harrassment does not need to be tolerated, and it can be an
enormous emotional support to discuss this with someone and know that
...
"""


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X) # Modifies count_vect
print(X_train_counts.shape) #(2257, 35788)

words = list(count_vect.vocabulary_.keys())
counts = list(count_vect.vocabulary_.values())
print(len(count_vect.vocabulary_)) # Dict of 35,788


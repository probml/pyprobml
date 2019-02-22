#https://scikit-learn.org/dev/tutorial/text_analytics/working_with_text_data.html

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)

X = data.data; # list of strings
Y = data.target; # numpy array of ints
print("Num. docs {}".format(len(X)))

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
print_data(data, 2)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X) # Modifies count_vect
print(X_train_counts.shape) #(2257, 35788)
print(len(count_vect.vocabulary_)) # Dict of 35,788

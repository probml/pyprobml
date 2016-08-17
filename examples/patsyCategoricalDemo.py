# demo of how patsy handles categorical variables

# Patsy notation is described here
#http://statsmode#ls.sourceforge.net/devel/example_formulas.html

#http://patsy.readthedocs.org/en/latest/categorical-coding.html
data = demo_data("a", nlevels=3)
dmatrix("a", data)
'''
DesignMatrix with shape (6, 3)
  Intercept  a[T.a2]  a[T.a3]
          1        0        0
          1        1        0
          1        0        1
          1        0        0
          1        1        0
          1        0        1
  Terms:
    'Intercept' (column 0)
    'a' (columns 1:3)
    '''
    
data = demo_data("a", nlevels=3)
dmatrix("a-1", data)
'''
DesignMatrix with shape (6, 3)
  a[a1]  a[a2]  a[a3]
      1      0      0
      0      1      0
      0      0      1
      1      0      0
      0      1      0
      0      0      1
  Terms:
    'a' (columns 0:3)'''
    
data = demo_data("a", nlevels=2)
dmatrix("a", data)
'''
DesignMatrix with shape (6, 2)
  Intercept  a[T.a2]
          1        0
          1        1
          1        0
          1        1
          1        0
          1        1
  Terms:
    'Intercept' (column 0)
    'a' (column 1)'''
    
data = demo_data("a", nlevels=2)
dmatrix("a-1", data)

'''
DesignMatrix with shape (6, 2)
  a[a1]  a[a2]
      1      0
      0      1
      1      0
      0      1
      1      0
      0      1
  Terms:
    'a' (columns 0:2)
    '''
    

# How to Contribute
Kevin Murphy and Mahmoud Soliman. 

**Last updated: 2021-06-03.**


We'd love to accept your patches and contributions to this project.
This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).

## Contributor License Agreement

Contributions to this project means that the contributors agree to releasing the contributions under the MIT license.

## Guidelines

Please follow the guidelines below when submitting code to [pyprobml](https://github.com/probml/pyprobml). In general, your commit should be in response to an open issue. Most of these issues currently concern converting legacy Matlab code to Python. In the future, we may create new issues related to converting Tensorflow or PyTorch code to JAX. 

### Github guidelines

- Follow standard github [guidelines](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/overview) on pull requests.
- To avoid duplication of work, please check the pull requests and issues to see if someone has already pushed a solution to the issue you are working on. If not, feel free to ‘claim’ it. However, do not submit multiple draft PRs that may block other people - work on one issue at a time.
- Please add [issue keyword](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword) on PRs. In particular, make sure you mention the issue it fixes. When the code is ready, request a review from both [@murphyk](https://github.com/murphyk) and [@mjsML](https://github.com/mjsML).
- Make sure your pull request has only one commit (We will squash the commits anyway, however github still emails all people following the repo for every single commit!).
- In general, your commit should be a single file. If you want to check in multiple files, discuss this in the thread for the github issue you are dealing with.
- If your pull request refers to an open issue, please be sure to mention it (e.g., 'Closes #foo') in your PR.
 
### Coding guidelines
Make sure your code works properly in Google's Colab. (It is not sufficient for it to work on your local machine). 
The first cell should contain the following boilerplate code, that emulates running locally:
```python
#!git clone https://github.com/probml/pyprobml /pyprobml &> /dev/null
#%cd -q /pyprobml/scripts
!mkdir figures
!mkdir scripts
%cd /content/scripts
!wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py
import pyprobml_utils as pml
```
You can then import any other libraries that your code needs, eg
```python
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math
import pandas as pd
import sklearn 
import scipy

import jax
import jax.numpy as jnp
from jax import random
```
If using Numpyro, you can use this:
```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4" # use 2 for regular colab, 4 for high memory (colab pro)
!pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro
import numpyro
import numpyro.distributions as dist
```
If using Pyro, you can use this:
```python
!pip3 install pyro-ppl
```
If you want to create local files within colab,
 you can use this idiom:
```python
file = 'kalman_tracking_demo.py' # change this filename as needed
!touch $file # create empty file
from google.colab import files
files.view(file) # open editor
```
Once it works, import it, or run it directly like so:
```python
%run $file  
```
To make sure colab notices any changes to the file, add this magic:
```
%load_ext autoreload
%autoreload 2
```
When it all works, it is generally best to 
just check in your file(s), not the notebook itself, unless the notebook has lots of explanatory text and figures, or needs a GPU or needs to install packages.
- Following the pep-8 guidlines, please name your python files using lowercase, with optional underscores separating words, as in foo_bar.py (even if this is different from the orignal matlab filename.) 
- Make sure your code reproduces the figure(s) in the book as closely as is “reasonable”. Note that in some cases things will not match exactly, e.g., because of different random number seeds. Do not worry about that, as long as your code is correct. Similarly, do not stress over small visual differences (e.g., colors or fonts), although the figure should be readable. 
- Follow the example below when  creating each figure (using the same figure file names as in the original Matlab code, if relevant).  For image filenames, use lowercase, but replace underscores with hyphens, as in foo-bar.pdf. 

```python
fig, ax = plt.subplots()
...
pml.savefig("test_figure.pdf") # will store in ../figures directory
plt.show() # this is necessary to force output  buffer to be flushed
```
- When labeling plots, please make sure you use [latex notation for math/ Greek symbols](https://matplotlib.org/stable/tutorials/text/mathtext.html), where possible.
- Please don't hardcode colors of your figure, use the default values. If you need to manually choose colors, use the [new default](https://matplotlib.org/stable/users/dflt_style_changes.html#colormap) colormap of matplotlib. This color map is designed to be viewable by color-blind people. If colors don't match the original (Matlab) figures, don't worry too much, as long as the logic is the same.
- Please use github public [gists](https://gist.github.com/) to share the figures that your code generates, so we can quickly “eyeball” them. Include these gists in your PR.
- Your implementation should match what is described in the book, but does not need to be identical to the original Matlab code (i.e., feel free to refactor things if it will improve your code).
- Follow standard Python style [guidelines](https://google.github.io/styleguide/pyguide.html#s3-python-style-rules). In particular, follow [PEP8 naming conventions](https://www.python.org/dev/peps/pep-0008/#function-and-variable-names).
- Avoid hard-coding [“magic numbers”](https://stackoverflow.com/questions/47882/what-is-a-magic-number-and-why-is-it-bad) in the body of your code. 
For example, instead of writing:

```python
xs = np.arange(0, 10)
ys = np.ones(10)
```
you should write something like this:
```python 
num_data = 10
xs = np.arange(0, num_data)
ys = np.ones(len(xs))
```
- Vectorize your code where possible. For example, instead of this slow doubly nested for loop:

```python
dist_mat = np.zeros((ndata, ndata))
for i in range(ndata):
  for j in range(ndata):
    dist_mat[i,j]  = np.sum((xdata[i,:] - xdata[j,:])**2)
``` 
You should write:


```python 
from scipy.spatial.distance import pdist, cdist
dist_mat = cdist(xdata, xdata, metric='sqeuclidean')
```

To access elements of an array in parallel, replace this
```python
X1 = []
for n in range(len(row)):
  i = row[n]
  j = col[n]
  X1.append(X[i,j])
```
with this
```python
X1 = X[row, col] # fancy indexing
```

To access a submatrix in parallel, replace this
```python
X2 = np.zeros([len(row), len(col)])
for itarget, isrc in enumerate(row):
  for jtarget, jsrc in enumerate(col):
    X2[itarget, jtarget]  = X[isrc, jsrc]
```
with this
```python
ndx = np.ix_(row, col)
X2 = X[ndx]
```

More details on Python's indexing can be found in [Jake Vanderplas's book](https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html).


For more advanced vectorization, consider using [JAX](https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/jax_intro.ipynb).
 
- Do not use JAX if you don’t need to, i.e., default to standard numpy and scipy, unless you need autograd or vmap or some other JAX features.
 
- Please make sure your code is reproducible by controlling randomness, eg use `np.random.state(0)` and `torch.manual_seed(0)`. (For JAX, use PRNG state.)
 


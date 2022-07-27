# How to Contribute
Kevin Murphy and Mahmoud Soliman. 

**Last updated: 2021-08-24.**


We'd love to accept your patches and contributions to this project.
This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).

## Contributor License Agreement

Contributions to this project means that the contributors agree to releasing the contributions under the MIT license.

## Guidelines

Please follow the guidelines below when submitting code to [pyprobml](https://github.com/probml/pyprobml). In general, your commit should be in response to an open issue. If you want to add something new, please open an issue first. 

### Github guidelines

- Follow standard github [guidelines](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/overview) on pull requests.
- To avoid duplication of work, please check the pull requests and issues to see if someone has already pushed a solution to the issue you are working on. If not, feel free to ‘claim’ it. However, do not submit multiple draft PRs that may block other people - work on one issue at a time.
- Please add [issue keyword](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword) on PRs. In particular, make sure you mention the issue it fixes. When the code is ready, request a review from both [@murphyk](https://github.com/murphyk) and [@mjsML](https://github.com/mjsML).
- Make sure your pull request has only one commit (We will squash the commits anyway, however github still emails all people following the repo for every single commit!).
- In general, your commit should be a single file. If you want to check in multiple files, discuss this in the thread for the github issue you are dealing with.
- If your pull request refers to an open issue, please be sure to mention it (e.g., 'Closes #foo') in your PR.
 


### Coding guidelines

- Make sure your code works properly in a Google Colab. (It is not sufficient for it to work on your local machine). 
Do not check in the notebook itself (unless you are creating a big tutorial with lots of text and pictures, in which case you should open a PR on https://github.com/probml/probml-notebooks).
- Use the
[superimport](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/Superimport.ipynb) library to install  all other dependencies.
- Your script file should contain the following noilerplate, at a minimum:
```python
import superimport
import pyprobml_utils as pml
import numpy as np
```
- Follow standard Python style [guidelines](https://google.github.io/styleguide/pyguide.html#s3-python-style-rules). In particular, follow [PEP8 naming conventions](https://www.python.org/dev/peps/pep-0008/#function-and-variable-names).
- Name your python files using lowercase, with optional underscores separating words, as in foo_bar.py (even if this is different from the orignal matlab filename.) 
- Follow the example below when  creating figures for the book (using the same file names as in the original Matlab code, if relevant).  
```python
fig, ax = plt.subplots()
...
pml.savefig("test_figure.pdf") # will store in ../figures directory
plt.show() # this is necessary to force output  buffer to be flushed
```
- When labeling plots, please make sure you use [latex notation for math/ Greek symbols](https://matplotlib.org/stable/tutorials/text/mathtext.html), where possible.
- Please don't hardcode colors of your figure, use the default values. If you need to manually choose colors, use the [new default](https://matplotlib.org/stable/users/dflt_style_changes.html#colormap) colormap of matplotlib. This color map is designed to be viewable by color-blind people. If colors don't match the original (Matlab) figures, don't worry too much, as long as the logic is the same.
- Please use github public [gists](https://gist.github.com/) to share the figures that your code generates, so we can quickly “eyeball” them. Include these gists in your PR.
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
 
- Use numpy for simple demos, JAX for when you need autograd or vmap, and pytorch for when you are reusing existing pytorch code. 
 
- Please make sure your code is reproducible by controlling randomness, eg use `np.random.state(0)` and `torch.manual_seed(0)`. (For JAX, use PRNG state.)
 


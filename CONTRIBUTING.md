# How to Contribute
Kevin Murphy and Mahmoud Soliman. 

**Last updated: 2022-03-10.**


We'd love to accept your patches and contributions to this project.
This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).

## Contributor License Agreement

Contributions to this project means that the contributors agree to releasing the contributions under the MIT license.

## Coding guidelines

- Follow standard Python style [guidelines](https://google.github.io/styleguide/pyguide.html#s3-python-style-rules). In particular, follow [PEP8 naming conventions](https://www.python.org/dev/peps/pep-0008/#function-and-variable-names).
- See [this list of python tutorials](https://github.com/probml/probml-notebooks/blob/main/markdown/python_tutorials.md) for more guidelines.
- Use [JAX](https://github.com/probml/probml-notebooks/blob/main/markdown/jax_tutorials.md)  instead of numpy/scipy.
- If you need to use a neural network, use [Flax](https://github.com/google/flax) instead of PyTorch/TF.
- In general it is best to write your code as a series of files, but **you should check that they can be run inside of [Google Colab](https://github.com/probml/probml-notebooks/blob/main/notebooks/colab_intro.ipynb)**, after installing necessary packages. 
Here is an example notebook to test your code (note that the main code lives in external files, we just use Colab (GPU mode) to run it from inside
a cell.)
```
# get your code
!git clone my_repo # link to your repo

# install common dependencies that are not already in colab
%%capture
%pip install --upgrade --user pip
%pip install --upgrade --user tensorflow tensorflow_probability
%pip install git+git://github.com/deepmind/optax.git
%pip install --upgrade git+https://github.com/google/flax.git
%pip install git+git://github.com/blackjax-devs/blackjax.git
%pip install git+git://github.com/deepmind/distrax.git
%pip install superimport  einops arviz


# get code for the book
!git clone https://github.com/probml/pyprobml
%pip install git+git://github.com/probml/jsl

# Test your code
%run my_repo/main.py
```
- If you want to check in a notebook,  open a PR on https://github.com/probml/probml-notebooks.
- A typical set of imports will look like this:
```
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sklearn
import flax
import flax.linen as nn
import einops
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from functools import partial
import tensorflow_datasets as tfds
import typing
import chex
import optax
```

## Github guidelines

- Follow standard github [guidelines](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/overview) on pull requests.
- In general, your commit should be in response to an open issue. If you want to add something new, please open an issue first. Some issues are quite large, so you may want to create smaller issues you can tackle individually, and  link these to the main one. 
- If you want to tackle an open issue, please reply to the issue to indicate your interest, and add a link to any design docs or partial solutions. 
- Do not submit multiple draft PRs - work on one issue at a time.
- If your pull request refers to an open issue, please be sure to mention it using the  [issue keyword](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword).
-  When the code is ready, request a review from  [@murphyk](https://github.com/murphyk), [@mjsML](https://github.com/mjsML) and 
[@gerdm](https://github.com/gerdm); one of us will get back to you within 1-2 days.
- Make sure your pull request has only one commit (We will squash the commits anyway, however github still emails all people following the repo for every single commit!).
- Please use github public [gists](https://gist.github.com/) to share the figures that your code generates, so we can quickly “eyeball” them.
 Include these gists in your PR.
- Look at [this example](https://github.com/probml/pyprobml/pull/690) for how to format your PR. 
- If your code is in a notebook stored in your forked copy of the repo, please include a link to it in your PR (similar to [this example](https://github.com/probml/pyprobml/pull/688)

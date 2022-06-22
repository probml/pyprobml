# JAX 


[JAX](https://github.com/google/jax) is a  version of NumPy that runs fast on CPU, GPU and TPU, by compiling down to XLA. It also has an excellent automatic differentiation library, extending the earlier [autograd](https://github.com/hips/autograd) package, which makes it easy to compute higher order derivatives, per-example gradients (instead of aggregated gradients), and gradients of complex code (e.g., optimize an optimizer).
The JAX interface is almost identical to NumPy (by design), but with some small differences, and lots of additional features.
 More details can be found in the other tutorials listed below.
 
 Here is a one-slide summary of JAX from a [recent 10 minute overview talk](https://youtu.be/WdTeDXsOSj4) by Jake Vanderplas.  
 <img src="https://github.com/probml/probml-notebooks/blob/main/images/jax-summary.png"
     alt="Jax summary slide"
     style="float: left; margin-right: 10px;" />

 

## Tutorials (blog posts / notebooks)

- [JAX homepage](https://github.com/google/jax)
- [JAX 101 (Deepmind tutorial)](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Thinking in JAX (Google tutorial)](https://colab.research.google.com/github/google/jax/blob/master/docs/notebooks/thinking_in_jax.ipynb)
- [List of common 'gotchas'](https://github.com/google/jax#current-gotchas)
- [Awesome JAX: extensive list of tutorials and code](https://github.com/n2cholas/awesome-jax)
- [Kevin Murphy's intro to JAX colab](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/jax_intro.ipynb)
- [Zeel Patel's JAX hints n tips](https://colab.research.google.com/github/patel-zeel/probml-notebooks/blob/jax_tricks/notebooks/Practical_JAX_Tips.ipynb)
-  [Haiku tutorial](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/haiku_intro.ipynb)
-  [Flax tutorial](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/flax_intro.ipynb)
- [From PyTorch to JAX: towards neural net frameworks that purify stateful code](https://sjmielke.com/jax-purify.htm). Explains concepts behind flax and haiku.
- [Getting started with JAX: MLPs, CNNs & RNNs](https://roberttlange.github.io/posts/2020/03/blog-post-10/)
- [Implement JAX in JAX](https://jax.readthedocs.io/en/latest/autodidax.html)
- [CMA-ES in JAX](https://roberttlange.github.io/posts/2021/02/cma-es-jax/) blog post for fitting DNNs using blackbox optimization.
- [JAX on TPU pods](http://matpalm.com/blog/ymxb_pod_slice/), solving y=mx+b with jax on a tpu pod slice by Mat Kelcey
- [Differentiable programming with JAX (by Eric Ma)](https://ericmjl.github.io/dl-workshop/)
- [JAX workshop from Aalto University, Nov. 2020](https://github.com/AdrienCorenflos/jax-workshop)

## Videos / talks

- [Matt Johnson's ACM talk, Nov 2020](https://www.youtube.com/watch?v=BzuEGdGHKjc)
- [JAX at Deepmind, NeurIPS 2020](https://www.youtube.com/watch?v=iDxJxIyzSiM)
- [Intro to JAX (25 min), Jake VanderPlas, SciPy2020](https://www.youtube.com/watch?v=z-WSrQDXkuM&t=6s)
- [Deepdive on JAX, Roy Frostig, Stanford MLSyst, Nov 2020](https://www.youtube.com/watch?v=mbUwCPiqZBM)
- [Intro to JAX (1h), Nicholas Vadivelu, Waterloo datascience course, July 2020](https://www.youtube.com/watch?v=QkmKfzxbCLQ&t=2583s). See also [his colab](https://github.com/n2cholas/dsc-workshops/blob/master/JAX_Demo.ipynb).

## JAX libraries related to ML

in this tutorial, we focus on core JAX.
However, since JAX is quite low level (like numpy), many libraries are being developed
that build on top of it, to provide more specialized functionality.
We summarize a few of the ML-related libraries below.
See also https://github.com/n2cholas/awesome-jax which has a more extensive list.

### DNN libraries

JAX is a purely functional library, which differs from Tensorflow and
Pytorch, which are stateful. The main advantages of functional programming
are that  we can safely transform the code, and/or run it in parallel, without worrying about
global state changing behind the scenes. The main disadvantage is that code (especially DNNs) can be harder to write.
To simplify the task, various DNN libraries have been designed, as we list below. In this book, we use Flax.

|Name|Description|
|----|----|
|[Stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)|Barebones library for specifying DNNs|
|[Flax](https://github.com/google/flax)|Library for specifying and training DNNs|
|[Haiku](https://github.com/deepmind/dm-haiku)|Library for specifying DNNs, similar to Sonnet|
|[Jraph](https://github.com/deepmind/jraph)| Library for graph neural networks|
|[Trax](https://github.com/google/trax)|Library for specifying and training DNNs, with a focus on sequence models|
|[T5X](https://github.com/google-research/google-research/tree/master/flax_models/t5x)|  T5 (a large seq2seq model) in JAX/Flax | 
|[HuggingFace Transformers](https://github.com/huggingface/transformers/tree/master/examples/flax) | Transformers |
|[Objax](https://github.com/google/objax)|PyTorch-like library for JAX (stateful/ object-oriented, not compatible with other JAX libraries)|
|[Elegy](https://github.com/poets-ai/elegy)|Keras-like library for Jax|
|[FlaxVision](https://github.com/rolandgvc/flaxvision)|Flax version of [torchvision](https://github.com/pytorch/vision)|
|[Neural tangents](https://github.com/google/neural-tangents)|Library to compute a kernel from a DNN|
|[Efficient nets](https://github.com/rwightman/efficientnet-jax)|Efficient CNN classifiers in Flax |
|[Progressive GANS](https://github.com/n2cholas/progan-flax)| Progressive GANs in Flax |



### RL libraries

|Name|Description|
|----|----|
|[RLax](https://github.com/deepmind/rlax)|Library from Deepmind|
|[Coax](https://github.com/microsoft/coax)|Lightweight library from Microsoft for solving Open-AI gym environments|

### Probabilistic programming languages


|Name|Description|
|----|----|
|[NumPyro](https://github.com/pyro-ppl/numpyro)|Library for PPL|
|[Oryx](https://github.com/tensorflow/probability/tree/master/spinoffs/oryx)|Lightweight library for PPL|
|[MCX](https://github.com/rlouf/mcx)| Library for PPL |
|[JXF](https://github.com/lindermanlab/jxf) | Exponential families in JAX|
|[Distrax](https://github.com/deepmind/distrax)| Library for probability distributions and bijectors|
|[TFP/JAX distributions](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions)| Jax port of tfp.distributions|
|[BlackJax](https://github.com/blackjax-devs/blackjax)| Library for HMC|
|[Newt](https://github.com/AaltoML/Newt/)| Variational inference for (Markov) GPs|

### Other libraries

There are also many other JAX libraries for tasks that are not about defining DNN models. We list some of them below.

|Name|Description|
|----|----|
|[Optax](https://github.com/deepmind/optax)|Library for defining gradient-based optimizers|
|[Chex](https://github.com/deepmind/chex)|Library for debugging and developing reliable JAX code|
|[Common loop utilities](https://github.com/google/CommonLoopUtils) |Library for writing "beautiful training loops in JAX"|
|[KF](https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers)| (Extended, Parallelized) Kalman filtering/ smoothing |
|[Easy Neural ODEs](https://github.com/jacobjinkelly/easy-neural-ode)| Neural ODEs for classification, Latent ODEs for time series and FFJORD for density estimation models with a bunch of higher order adaptive-stepping numercial solvers(e.g. Heun-Euler, Fehlberg,Cash-Karp,Tanyam and Adams adaptive order)|
|[GTP-J](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/)| Open source version of GPT-3 using JAX and TPU v3-256.|
|[CLIP-Jax](https://github.com/kingoflolz/CLIP_JAX) | Jax wrapper for inference using pre-trained OpenAI CLIP model |
|[Scenic](https://github.com/google-research/scenic])| Computer vision library |
|[Pix](https://github.com/deepmind/dm_pix)| Image pre-processing library|


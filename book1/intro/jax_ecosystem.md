# JAX ecosystem

[JAX](http://github.com/google/jax) is basically a fast version of
numpy, combined with autograd. (The speed comes from various program
transformations, such as a Just In Time compiler which converts the code to XLA,
which runs on various  hardware accelerators, such as GPUs and TPUs.)

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
|[Jraph](https://github.com/deepmind/jraph)| Library for grapn neural networks|
|[Trax](https://github.com/google/trax)|Library for specifying and training DNNs, with a focus on sequence models|
|[Objax](https://github.com/google/objax)|Stateful (object-oriented) DNN framework, similar to PyTorch, not compatible with other JAX libraries|

There are also many other JAX libraries for tasks that are not about defining DNN models. We list some of them below.

|Name|Description|
|----|----|
|[NumPyro](https://github.com/pyro-ppl/numpyro)|Library for (deep) probabilistic modeling|
|[Optax](https://github.com/deepmind/optax)|Library for defining gradient-based optimizers|
|[RLax](https://github.com/deepmind/rlax)|Library for reinforcement learning|
|[Chex](https://github.com/deepmind/chex)|Library for debugging and developing reliable JAX code|



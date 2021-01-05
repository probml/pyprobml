# Software for deep learning <a class="anchor" id="DL"></a>


Deep learning is about composing differentiable functions into more complex functions, represented as a computation graph, and then using automatic differentiation ("autograd") to compute gradients, which we can pass to an optimizer, to fit the function to data. This is sometimes called "differentiable programming".

There are several libraries that can execute such computation graphs on hardware accelerators, such as GPUs. (Some libraries also support distributed computation, but we will not need use this feature in this book.) We list a few popular libraries below.

|Name|More info|
|----|----|
|[Tensorflow2](http://www.tensorflow.org)|[tf_intro.ipynb](https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/tf_intro.ipynb)
|[JAX](http://github.com/google/jax)|[jax_intro.ipynb](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/jax.ipynb)
|[PyTorch](http://pytorch.org)|[PyTorch website](https://pytorch.org/tutorials)
|[MXNet](https://mxnet.apache.org)|[Dive into deep learning book](http://www.d2l.ai)


# Multilayer perceptrons (MLPs)

|Title|Software|Link|
|-----------|----|----|
|Auto-MPG regression|TF2|[TF2 tutorials](https://www.tensorflow.org/tutorials/keras/regression)
|Tabular medical data classification|TF2|[TF2 tutorials](https://www.tensorflow.org/tutorials/structured_data/feature_columns)
|(Fashion) MNIST image classification|TF2|[mlp_mnist_tf.ipynb](mlp_mnist_tf.ipynb)
|IMDB movie review sentiment classification |TF2|[mlp_imdb_tf.ipynb](mlp_imdb_tf.ipynb)
|IMDB movie review sentiment classification using pre-trained word embeddings from TF-hub|TF2|[TF2 tutorials](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)
|IMDB movie review sentiment classification using keras pre-processed data|TF2|[TF2 tutorials](https://www.tensorflow.org/tutorials/keras/text_classification)|
|Heteroskedastic regression in 1d| TFP | [mlp_1d_regression_hetero_tf.ipynb](mlp_1d_regression_hetero_tf.ipynb)|
|Using tensorfboard to plot learning curves| TF2 | [early_stopping_tensorboard_tf.ipynb](early_stopping_tensorboard_tf.ipynb)
|Hierarchical Bayes for BNNs| PyMC3 | [bnn_hierarchical_pymc3.ipynb](bnn_hierarchical_pymc3.ipynb)



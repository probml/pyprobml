## Software for deep learning <a class="anchor" id="DL"></a>


Deep learning is about composing differentiable functions into more complex functions, represented as a computation graph, and then using automatic differentiation ("autograd") to compute gradients, which we can pass to an optimizer, to fit the function to data. This is sometimes called "differentiable programming".

There are several libraries that can execute such computation graphs on hardware accelerators, such as GPUs. (Some libraries also support distributed computation, but we will not need use this feature in this book.) We list a few popular libraries below.



<table align="left">
    <tr>
        <th style="text-align:left">Name</th>
        <th style="text-align:left" width="400">Functionality</th>
      <th style="text-align:left">More info</th>
    <tr> 
        <td style="text-align:left"> <a href="http://www.tensorflow.org">Tensorflow 2</a></td>
            <td style="text-align:left"> Accelerated numpy-like library with autograd support. Keras API.</td>
     <td style="text-align:left"><a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/tf.ipynb">TF notebook</a>
               <tr>
        <td style="text-align:left"> <a href="http://github.com/google/jax">JAX</a></td>
        <td style="text-align:left">Accelerated numpy, functional code transformations (autograd, JIT, VMAP, etc)</td>
            <td style="text-align:left">
              <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/jax.ipynb">JAX notebook</a>
    <tr>
        <td style="text-align:left"> <a href="http://pytorch.org">Pytorch</a></td>
         <td style="text-align:left"> Similar to TF 2</td>
       <td style="text-align:left">
       <a href="https://pytorch.org/tutorials/">Official PyTorch tutorials</a>
              <tr>
        <td style="text-align:left"> <a href="https://mxnet.apache.org/">MXNet</a>
            <td  style="text-align:left"> Similar to TF 2. Gluon  API.
              <td style="text-align:left">
                <a href="http://www.d2l.ai/">  Dive into deep learning book</a>       
</table>

# Multilayer perceptrons (MLPs)

In this section, we give some examples of MLPs using TF2.
(* denotes official TF tutorials, which are not part of the pyprobml repo.)

* [Auto-MPG regression *](https://www.tensorflow.org/tutorials/keras/regression)
* [Tabular medical data classification *](https://www.tensorflow.org/tutorials/structured_data/feature_columns)
*  [FashionMNIST image classification](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/dnn/fashion_mlp_tf.ipynb) 
* [FashionMNIST image classification *](https://www.tensorflow.org/tutorials/keras/classification)
* [IMDB movie review sentiment classification](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/dnn/imdb_mlp_tf.ipynb) 
* [IMDB movie review sentiment classification using pre-trained word embeddings from TF-hub *](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)
* [IMDB movie review sentiment classification using keras pre-processed data *](https://www.tensorflow.org/tutorials/keras/text_classification)


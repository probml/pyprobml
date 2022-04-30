# Tips and tricks on using Colab, Google Cloud Platform and TPUs

Authors: [murphyk](https://github.com/murphyk), [mjsML](https://github.com/mjsML), [gerdm](https://github.com/gerdm), summer  2021.

## Links

* [Colab tutorial](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/colab_intro.ipynb)
* [Using Google Cloud Storage from Colab](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/GCS_demo_v2.ipynb)
* [Using TPU VMs from Colab](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/tpu_colab_tutorial.ipynb)
* [Accessing colab machine via ssh](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/ssh_tunnels_and_how_to_dig_them.ipynb)
* [Hugging face tutorial page on TPU VMs](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md#how-to-setup-tpu-vm)
* [Skye's tutorial video on TPU VMs](https://www.youtube.com/watch?v=fuAyUQcVzTY)
* [Using GCP/ TPU VMs via SSH + VSCode](https://github.com/probml/probml-notebooks/blob/main/markdown/gcp_ssh_vscode.md)
* [Using tensorboard](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/tbdev_getting_started.ipynb)

## Random stuff

* [Screenshot of how to split your v3-8 TPU VM into two 4-core machines](https://github.com/probml/probml-notebooks/blob/main/images/jax-tpu-split.png). This lets you run two separate JAX processes at once, in different jupyter notebooks (eg if working on two projects simultaneously). The magic incantation is from [this gist](https://gist.github.com/skye/f82ba45d2445bb19d53545538754f9a3) written by Skye Wanderman-Milne and is reproduced below.
```
# 2x 2 chips (4 cores) per process:
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
# Different per process:
os.environ["TPU_VISIBLE_DEVICES"] = "0,1" # Change to "2,3" for the second machine
```

* [Python multiprocessing library](https://docs.python.org/3/library/multiprocessing.html)
* [Jax on CPUs](https://github.com/google/jax/issues/1598#issuecomment-548031576)
* [pmap on CPUs](https://github.com/google/jax/issues/1408)
* `pip install git+https://github.com/blackjax-devs/blackjax`

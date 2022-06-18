# pyprobml

<img src="https://img.shields.io/github/stars/probml/pyprobml?style=social">


Python 3 code to reproduce the figures in the book series [Probabilistic Machine Learning](https://probml.github.io/pml-book/) by Kevin Patrick Murphy.
This is work in progress, so expect rough edges!
(Some demos use code from our companion [JAX State Space Library](https://github.com/probml/JSL).)
 

## Running the notebooks

The scripts needed to make all the figures for each chapter are automatically combined together into a series of Jupyter notebooks, one per chapter.
* [Volume 1 figure notebooks](https://github.com/probml/pml-book/tree/main/pml1/)
* [Volume 2 figure notebooks](https://github.com/probml/pml-book/tree/main/pml2). (Note: volume 2 is not finished yet.)

In addition to the automatically generated notebooks, there are a series of manually created notebooks, which create additional figures, and provide supplementary material for the book. These are stored in the [notebooks repo](https://github.com/probml/probml-notebooks), since they can be quite large. Some of these notebooks use the scripts mentioned above, but others are independent of the book content.

The easiest way to run these notebooks is inside [Colab](https://colab.research.google.com/notebooks/intro.ipynb). This has most of the libraries you will need (e.g., scikit-learn,  JAX) pre-installed, and gives you access to a free GPU and TPU. We have a created a [intro to colab](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/colab_intro.ipynb) notebook with more details.


## Running scripts in colab

The easiest way to run individual scripts is inside [Colab](https://colab.research.google.com/notebooks/intro.ipynb). 
Just cut and paste this into a code cell:
```py
%pip install superimport 
!git clone --depth 1 https://github.com/probml/pyprobml  &> /dev/null # THIS CODEBASE
```
Note: The [superimport](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/Superimport.ipynb)
library will automatically install packages for any file which contains the line `import superimport'.

Then run a script from a cell like this:
```py
%run pyprobml/scripts/softmax_plot.py
```


To edit a file locally and then run it, follow the example below.
```py
# Make sure local changes to file are detected by runtime
%load_ext autoreload
%autoreload 2

file = 'pyprobml/scripts/softmax_plot.py' # change this filename as needed
from google.colab import files
files.view(file) # open editor

%run $file
```

To download and run code from github, follow the example below.
(Note the `raw` in the URL.)
```py
!wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/softmax_plot.py
%run softmax_plot.py
```

## Running the scripts locally 

We assume you have already installed [JAX](https://github.com/google/jax#installation) and
[Tensorflow](https://www.tensorflow.org/install) and [Torch](https://pytorch.org/),
since the details on how to do this depend on whether you have a CPU, GPU, etc.

For the remaining python packages, do this:
```bash
pip install superimport 
git clone --depth 1 https://github.com/probml/pyprobml  &> /dev/null # THIS CODEBASE
```

Note: The [superimport](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/Superimport.ipynb)
library will automatically install packages for any file which contains the line `import superimport'. 


To manually execute an individual script from the command line,
follow this example:
```bash
python3 pyprobml/scripts/softmax_plot.py 
```
This will  run the script, plot a figure, and save the result to the `pyprobml/figures` directory.

## Running scripts for vol 2

Some demos for vol 2 use [JSL (Jax State-space Library)](https://github.com/probml/JSL).
This requires extra packages, see [these installation instructions](https://github.com/probml/JSL#installation).
Then you can run the pyprobml version of the JSL demos like this
```
%run pyprobml/scripts/kf_tracking_demo.py # colab
python3 pyprobml/scripts/kf_tracking_demo.py # locally
```


## GCP, TPUs, and all that

When you want more power or control than colab gives you, you should get a Google Cloud Platform (GCP) account, and get access to a TPU VM. You can then use this as a virtual desktop which you can access via ssh from inside VScode. We have created [various tutorials on Colab, GCP and TPUs](https://github.com/probml/probml-notebooks/blob/main/markdown/colab_gcp_tpu_tutorial.md) with more information.


## How to contribute

See [this guide](https://github.com/probml/pyprobml/blob/master/CONTRIBUTING.md) for how to contribute code.


## Metrics

[![Stargazers over time](https://starchart.cc/probml/pyprobml.svg)](https://starchart.cc/probml/pyprobml)

## GSOC 2021

For a summary of some of the contributions to this codebase during Google Summer of Code 2021,
see [this link](https://probml.github.io/pml-book/gsoc2021.html).




<h2><a id="acknowledgements"></a>Acknowledgements</h2>

I would like to thank the following people for contributing to the code
(list autogenerated from [this page](https://thodorisbais.github.io/markdown-contributors/)):

[<img alt="murphyk" src="https://avatars.githubusercontent.com/u/4632336?v=4&s=117 width=117">](https://github.com/murphyk) |[<img alt="mjsML" src="https://avatars.githubusercontent.com/u/7131192?v=4&s=117 width=117">](https://github.com/mjsML) |[<img alt="Drishttii" src="https://avatars.githubusercontent.com/u/35187749?v=4&s=117 width=117">](https://github.com/Drishttii) |[<img alt="Duane321" src="https://avatars.githubusercontent.com/u/19956442?v=4&s=117 width=117">](https://github.com/Duane321) |[<img alt="gerdm" src="https://avatars.githubusercontent.com/u/4108759?v=4&s=117 width=117">](https://github.com/gerdm) |[<img alt="animesh-007" src="https://avatars.githubusercontent.com/u/53366877?v=4&s=117 width=117">](https://github.com/animesh-007) |[<img alt="Nirzu97" src="https://avatars.githubusercontent.com/u/28842790?v=4&s=117 width=117">](https://github.com/Nirzu97) |[<img alt="always-newbie161" src="https://avatars.githubusercontent.com/u/66471669?v=4&s=117 width=117">](https://github.com/always-newbie161) |[<img alt="karalleyna" src="https://avatars.githubusercontent.com/u/36455180?v=4&s=117 width=117">](https://github.com/karalleyna) |[<img alt="nappaillav" src="https://avatars.githubusercontent.com/u/43855961?v=4&s=117 width=117">](https://github.com/nappaillav) |[<img alt="jdf22" src="https://avatars.githubusercontent.com/u/1637094?v=4&s=117 width=117">](https://github.com/jdf22) |[<img alt="shivaditya-meduri" src="https://avatars.githubusercontent.com/u/77324692?v=4&s=117 width=117">](https://github.com/shivaditya-meduri) |[<img alt="Neoanarika" src="https://avatars.githubusercontent.com/u/5188337?v=4&s=117 width=117">](https://github.com/Neoanarika) |[<img alt="andrewnc" src="https://avatars.githubusercontent.com/u/7716402?v=4&s=117 width=117">](https://github.com/andrewnc) |[<img alt="Abdelrahman350" src="https://avatars.githubusercontent.com/u/47902062?v=4&s=117 width=117">](https://github.com/Abdelrahman350) |[<img alt="Garvit9000c" src="https://avatars.githubusercontent.com/u/68856476?v=4&s=117 width=117">](https://github.com/Garvit9000c) |[<img alt="kzymgch" src="https://avatars.githubusercontent.com/u/10054419?v=4&s=117 width=117">](https://github.com/kzymgch) |[<img alt="alen1010" src="https://avatars.githubusercontent.com/u/42214173?v=4&s=117 width=117">](https://github.com/alen1010) |[<img alt="adamnemecek" src="https://avatars.githubusercontent.com/u/182415?v=4&s=117 width=117">](https://github.com/adamnemecek) |[<img alt="galv" src="https://avatars.githubusercontent.com/u/4767568?v=4&s=117 width=117">](https://github.com/galv) |[<img alt="krasserm" src="https://avatars.githubusercontent.com/u/202907?v=4&s=117 width=117">](https://github.com/krasserm) |[<img alt="nealmcb" src="https://avatars.githubusercontent.com/u/119472?v=4&s=117 width=117">](https://github.com/nealmcb) |[<img alt="petercerno" src="https://avatars.githubusercontent.com/u/1649209?v=4&s=117 width=117">](https://github.com/petercerno) |[<img alt="Prahitha" src="https://avatars.githubusercontent.com/u/44160152?v=4&s=117 width=117">](https://github.com/Prahitha) |[<img alt="khanshehjad" src="https://avatars.githubusercontent.com/u/31896767?v=4&s=117 width=117">](https://github.com/khanshehjad) |[<img alt="hieuza" src="https://avatars.githubusercontent.com/u/1021144?v=4&s=117 width=117">](https://github.com/hieuza) |[<img alt="jlh2018" src="https://avatars.githubusercontent.com/u/40842099?v=4&s=117 width=117">](https://github.com/jlh2018) |[<img alt="mvervuurt" src="https://avatars.githubusercontent.com/u/6399881?v=4&s=117 width=117">](https://github.com/mvervuurt) |[<img alt="TripleTop" src="https://avatars.githubusercontent.com/u/48208522?v=4&s=117 width=117">](https://github.com/TripleTop) |
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
[murphyk](https://github.com/murphyk)|[mjsML](https://github.com/mjsML)|[Drishttii](https://github.com/Drishttii)|[Duane321](https://github.com/Duane321)|[gerdm](https://github.com/gerdm)|[animesh-007](https://github.com/animesh-007)|[Nirzu97](https://github.com/Nirzu97)|[always-newbie161](https://github.com/always-newbie161)|[karalleyna](https://github.com/karalleyna)|[nappaillav](https://github.com/nappaillav)|[jdf22](https://github.com/jdf22)|[shivaditya-meduri](https://github.com/shivaditya-meduri)|[Neoanarika](https://github.com/Neoanarika)|[andrewnc](https://github.com/andrewnc)|[Abdelrahman350](https://github.com/Abdelrahman350)|[Garvit9000c](https://github.com/Garvit9000c)|[kzymgch](https://github.com/kzymgch)|[alen1010](https://github.com/alen1010)|[adamnemecek](https://github.com/adamnemecek)|[galv](https://github.com/galv)|[krasserm](https://github.com/krasserm)|[nealmcb](https://github.com/nealmcb)|[petercerno](https://github.com/petercerno)|[Prahitha](https://github.com/Prahitha)|[khanshehjad](https://github.com/khanshehjad)|[hieuza](https://github.com/hieuza)|[jlh2018](https://github.com/jlh2018)|[mvervuurt](https://github.com/mvervuurt)|[TripleTop](https://github.com/TripleTop)|

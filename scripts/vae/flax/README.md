# Flax Implementation of VAEs

## To run mnist
1. install packages
```
pip install pytorch-lightning flax
```
2. Run the script
- In a jupyter notebook cell
```
%run vae_conv_mnist_flax_lib.py
```
- In a command line/terminal
```
python vae_conv_mnist_flax_main.py
```
## To run celeba
### Donwload data from kaggle
First make sure you have kaggle.json, instructions for downloading kaggle.json can be found [here](), then run the following commands in a cell or terminal 
```
mkdir /root/.kaggle
cp kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
rm kaggle.json
python download_celeba.py 
```
Then to download the data first donwload the following [script](https://github.com/probml/pyprobml/blob/master/scripts/download_celeba.py)
```
wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/download_celeba.py
```
### To run the  script
1. install packages
```
pip install pytorch-lightning flax
```
2. Run the script
- In a jupyter notebook cell
```
%run vae_conv_celeba_flax_lib.py
```
- In a command line/terminal
```
python vae_conv_mnist_flax_main.py
```
## Example notebooks: 
- Celeba dataset: <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/scripts/vae/flax/flax_conv_vae_celeba_notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

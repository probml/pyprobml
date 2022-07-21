<h1 align="center">
  <b>PyProbML GAN zoo 🐅 </b><br>
</h1>

This is a collection of Generative Adverserial Networks (GANs) implemented in *pytorch* written by Ang Ming Liang (Neoanarika@).
The  focus is on reproducibility and creating reusable blocks that can be used in any project. The aim of this project is to provide
a quick and simple working example for many of the cool GAN idea in the textbook. All the models are trained on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for consistency and comparison. 


Compare results of different GANs : 
<a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/gan_compare_results.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


GAN tricks and what the different GAN try to address :
<a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/gan_tricks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Requirements
- Python >= 3.7
- PyTorch >= 1.8
- Pytorch Lightning  == 1.4.0
- CUDA enabled computing device

## To download this subdirectory only 

Copy the url of the subdirectory and past it to this [webstie](https://download-directory.github.io) and then download this subdirectory as a zipfile

## Instruction For Training The Model

### Download celeba data

**Important :** Make sure to get your kaggle.json from [these instructions](https://github.com/Kaggle/kaggle-api#api-credentials) then run 

```
mkdir /root/.kaggle 
cp kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
rm kaggle.json
```

to copy kaggle.json into a folder first. Then to download the data first donwload the following [script](https://github.com/probml/pyprobml/blob/master/scripts/download_celeba.py)
```
wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/download_celeba.py
```
and run the following script
```
python download_celeba.py
```

### To Train Model

```
python run.py -config ./configs/dcgan.yaml
```
## Results

| Model                                                                  | Paper                                            | Samples |
|------------------------------------------------------------------------|--------------------------------------------------|---------|
| GAN ([Code][dcgan_code], [Config][dcgan_config])                       |[Link](https://arxiv.org/abs/1406.2661)           |  ![][1] |
| SNGAN ([Code][sngan_code], [Config][sngan_config])                     |[Link](https://arxiv.org/abs/1802.05957)          |  ![][3] |
| LOGAN ([Code][logan_code], [Config][logan_config])                     |[Link](https://arxiv.org/abs/1912.00953)          |  ![][2] |
| WGAN ([Code][wgan_code], [Config][wgan_config])                        |[Link](https://arxiv.org/abs/1701.07875)          |  ![][4] |
| GP-WGAN   ([Code][gp_wgan_code], [Config][gp_wgan_config])             |[Link](https://arxiv.org/pdf/1704.00028.pdf)      |  ![][5] |


## Acknowledgement

The idea of this zoo and some of the scripts were based on Anand Krishnamoorthy [Pytorch-VAE library](https://github.com/AntixK/PyTorch-VAE), we also used the script from [sayantanauddy](https://github.com/sayantanauddy/vae_lightning) to transform and download the celeba from kaggle. 

-----------

[dcgan_code]: https://raw.githubusercontent.com/probml/pyprobml/master/gan/models/dcgan.py
[gp_wgan_code]: https://raw.githubusercontent.com/probml/pyprobml/master/gan/models/gp_wgan.py
[logan_code]: https://raw.githubusercontent.com/probml/pyprobml/master/gan/models/logan.py
[sngan_code]: https://raw.githubusercontent.com/probml/pyprobml/master/gan/models/sngan.py
[wgan_code]: https://github.com/probml/pyprobml/blob/master/gan/models/wgan.py

[dcgan_config]: https://github.com/probml/pyprobml/blob/master/gan/configs/dcgan.yaml
[gp_wgan_config]: https://github.com/probml/pyprobml/blob/master/gan/configs/gp_wgan.yaml
[logan_config]: https://github.com/probml/pyprobml/blob/master/gan/configs/logan.yaml
[sngan_config]: https://github.com/probml/pyprobml/blob/master/gan/configs/sngan.yaml
[wgan_config]: https://github.com/probml/pyprobml/blob/master/gan/configs/wgan.yaml

[1]: https://github.com/probml/pyprobml/blob/master/gan/assets/dcgan.png
[2]: https://github.com/probml/pyprobml/blob/master/gan/assets/logan.png
[3]: https://github.com/probml/pyprobml/blob/master/gan/assets/sngan.png
[4]: https://github.com/probml/pyprobml/blob/master/gan/assets/wgan.png
[5]: https://github.com/probml/pyprobml/blob/master/gan/assets/gp_wgan.png

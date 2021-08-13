<h1 align="center">
  <b>PyProbML VAE zoo 🐘 </b><br>
</h1>

Compare results of different VAEs : <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/scripts/vae/compare_results.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

VAE tricks and what the different VAE try to address : <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/scripts/vae/vae_tricks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A collection of Variational AutoEncoders (VAEs) implemented in pytorch with focus on reproducibility and creating reusable blocks that can be used in any project. The aim of this project is to provide
a quick and simple working example for many of the cool VAE idea in the textbook. All the models are trained on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
for consistency and comparison. 

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

to copy kaggle.json into a folder first. 

### To Train Model

```
python run.py -config ./configs/vanilla_vae.yaml
```

## Results

| Model                                                                  | Paper                                            |Reconstruction | Samples |
|------------------------------------------------------------------------|--------------------------------------------------|---------------|---------|
| Original Images (for reconstruction)                                   |**N/A**                                           |    ![][1]     | **N/A** |
| AE ([Code][ae_code], [Config][ae_config])                              |**N/A**                                           |    ![][2]     | ![][10] |
| VAE ([Code][vae_code], [Config][vae_config])                           |[Link](https://arxiv.org/abs/1312.6114)           |    ![][2]     | ![][10] |
| Hinge VAE ([Code][hingevae_code], [Config][hingevae_config])           |[Link](https://arxiv.org/abs/1606.04934)          |    ![][3]     | ![][11] |
| MMD VAE ([Code][mmdvae_code], [Config][mmdvae_config])                 |[Link](https://arxiv.org/abs/1706.02262)          |    ![][4]     | ![][12] |
| Info VAE   ([Code][infovae_code], [Config][infovae_config])            |[Link](https://arxiv.org/abs/1706.02262)          |    ![][5]     | ![][13] |
| LogCosh VAE   ([Code][logcoshvae_code], [Config][logcoshvae_config])   |[Link](https://openreview.net/forum?id=rkglvsC9Ym)|    ![][6]     | ![][14] |
| Two-stage VAE   ([Code][twostage_code], [Config][twostage_config])     |[Link](https://arxiv.org/abs/1903.05789)          |    ![][7]     | ![][15] |
| Sigma VAE   ([Code][sigma_code], [Config][sigma_config])               |[Link](https://arxiv.org/abs/2006.13202)          |    ![][8]     | ![][16] |
| VQ-VAE (*K = 512, D = 64*) ([Code][vqvae_code], [Config][vqvae_config]) + PixelCNN([Code][pixelCNN_code]) |[Link](https://arxiv.org/abs/1711.00937)          |    ![][9]     | ![][17] |

-----------

[ae_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/vanilla_ae.py
[vae_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/vanilla_vae.py
[mmdvae_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/mmd_vae.py
[hingevae_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/hinge_vae.py
[logcoshvae_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/logcosh_vae.py
[infovae_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/info_vae.py
[vqvae_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/vq_vae.py
[twostage_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/two_stage_vae.py
[sigma_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/sigma_vae.py
[pixelCNN_code]: https://github.com/probml/pyprobml/blob/master/scripts/vae/models/sigma_vae.py

[ae_config]: https://github.com/probml/pyprobml/blob/master/scripts/vae/configs/vanilla_ae.yaml
[vae_config]: https://github.com/probml/pyprobml/blob/master/scripts/vae/configs/vanilla_vae.yaml
[logcoshvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/logcosh_vae.yaml
[infovae_config]: https://github.com/probml/pyprobml/blob/master/scripts/vae/configs/info_vae.yaml
[vqvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/vq_vae.yaml
[mmdvae_config]: https://github.com/probml/pyprobml/blob/master/scripts/vae/configs/mmd_vae.yaml
[hingevae_config]: https://github.com/probml/pyprobml/blob/master/scripts/vae/configs/hinge_vae.yaml
[twostage_config]: https://github.com/probml/pyprobml/blob/master/scripts/vae/configs/two_stage_vae.yaml
[sigma_config]: https://github.com/probml/pyprobml/blob/master/scripts/vae/configs/sigma_vae.yaml

[1]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/original.png
[2]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/vanilla_vae_recon.png
[3]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/hinge_vae_recon.png
[4]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/mmd_vae_recon.png
[5]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/info_vae_recon.png
[6]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/logcosh_vae_recon.png
[7]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/two_stage_vae_recon.png
[8]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/sigma_vae_recon.png
[9]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/vq_vae_recon.png
[10]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/vanilla_vae_samples.png
[11]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/hinge_vae_samples.png
[12]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/mmd_vae_samples.png
[13]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/info_vae_samples.png
[14]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/logcosh_vae_samples.png
[15]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/two_stage_vae_samples.png
[16]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/sigma_vae_samples.png
[17]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/vq_vae_samples.png
[18]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/vanilla_vae_recon.png
[19]: https://github.com/probml/pyprobml/blob/master/scripts/vae/assets/vanilla_vae_samples.png

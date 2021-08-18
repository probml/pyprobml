import torch 
import argparse
from assembler import get_config, assembler
from download_celeba import celeba_dataloader
from pytorch_lightning import Trainer
import torchvision.transforms as transforms

# Load configs
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
config = get_config(args.filename)
vae = assembler(config, "training")

# Load data
dm = celeba_dataloader(config["exp_params"]["batch_size"],
                       config["exp_params"]["img_size"],
                       config["exp_params"]["crop_size"],
                        config["exp_params"]["data_path"])


# Run Training Loop
trainer= Trainer(gpus = config["trainer_params"]["gpus"],
        max_epochs = config["trainer_params"]["max_epochs"])
trainer.fit(vae, datamodule=dm)
torch.save(vae.state_dict(), f"{vae.model.name}_celeba_conv.ckpt")

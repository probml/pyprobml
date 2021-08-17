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
gan = assembler(config)

# Load data
dm = celeba_dataloader(config["exp_params"]["batch_size"],
                       config["exp_params"]["img_size"],
                       config["exp_params"]["crop_size"],
                        config["exp_params"]["data_path"])


# Run Training Loop
trainer= Trainer(gpus = config["trainer_params"]["gpus"],
        max_epochs = config["trainer_params"]["max_epochs"])
trainer.fit(gan, datamodule=dm)
torch.save(gan.state_dict(), f"{gan.name}_celeba.ckpt")

import torch 
import argparse
from assembler import get_config, assembler
from data import  CelebADataModule
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
trans = []
trans.append(transforms.RandomHorizontalFlip())
if config["exp_params"]["crop_size"] > 0:
  trans.append(transforms.CenterCrop(config["exp_params"]["crop_size"]))
trans.append(transforms.Resize(config["exp_params"]["img_size"]))
trans.append(transforms.ToTensor())
transform = transforms.Compose(trans)

dm = CelebADataModule(data_dir=config["exp_params"]["data_path"],
                                target_type='attr',
                                train_transform=transform,
                                val_transform=transform,
                                download=True,
                                batch_size=config["exp_params"]["batch_size"])

# Run Training Loop
trainer= Trainer(gpus = config["trainer_params"]["gpus"],
        max_epochs = config["trainer_params"]["max_epochs"])
trainer.fit(gan, datamodule=dm)
torch.save(gan.state_dict(), f"{gan.name}_celeba.ckpt")
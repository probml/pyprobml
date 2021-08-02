import yaml
import torch 
import argparse
from assembler import assembler
from experiment import VAEModule
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
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Assemble model
model_name = config["exp_params"]["model_name"]
vae = assembler(model_name , config)
m = VAEModule(vae, config["exp_params"]["LR"])

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
trainer.fit(m, datamodule=dm)
torch.save(m.state_dict(), f"{model_name}-celeba-conv.ckpt")
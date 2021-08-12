import torch
import argparse
from assembler import get_config, assembler
from data import  CelebADataModule
from pytorch_lightning import Trainer
import torchvision.transforms as transforms

# Load configs
from models.pixel_cnn import *
from experiment import *

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


vae.load_state_dict(torch.load(config["pixel_params"]["pretrained_path"]))

num_residual_blocks = config["pixel_params"]["num_residual_blocks"]
num_pixelcnn_layers = config["pixel_params"]["num_pixelcnn_layers"]
num_embeddings = config["vq_params"]["num_embeddings"]
hidden_dim = config["pixel_params"]["hidden_dim"]

# Run Training Loop
trainer= Trainer(gpus = config["trainer_params"]["gpus"],
                max_epochs = config["trainer_params"]["max_epochs"])

pixel_cnn_raw = PixelCNN(hidden_dim, num_residual_blocks, num_pixelcnn_layers, num_embeddings)
pixel_cnn = PixelCNNModule(pixel_cnn_raw,
                           vae,
                           config["pixel_params"]["height"],
                           config["pixel_params"]["width"],
                           config["pixel_params"]["LR"])
trainer.fit(pixel_cnn, datamodule=dm)
pixel_cnn.save(config["pixel_params"]["save_path"])
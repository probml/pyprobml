import yaml
import importlib
from functools import partial
from models.base_gan import GAN

def get_config(fpath):
   with open(fpath, 'r') as file:
      try:
          config = yaml.safe_load(file)
      except yaml.YAMLError as exc:
          print(exc)
      return config

def is_config_valid(config):
    assert config["loss_params"]["latent_dim"] == config["generator_params"]["latent_dim"]

def assembler(config):
    # Get model name
    is_config_valid(config)

    # Get model components 
    gan_name = config["exp_params"]["model_name"]
    componets = importlib.import_module(f"models.{gan_name}")
    discriminator = componets.Discriminator(**config["discriminator_params"])
    generator = componets.Generator(**config["generator_params"])
    disc_loss = partial(componets.disc_loss, config)
    gen_loss = partial(componets.gen_loss, config)

    # Get sampling components 
    sampling_name = config["exp_params"]["refinement"]
    if sampling_name is not None:
        sampler = importlib.import_module(f"sampling.{sampling_name}")
        sampling = lambda x : sampler.sampling(config["sampling_params"], generator, discriminator, x)
    else:
        sampling = None 
    gan = GAN(gan_name, generator, discriminator, gen_loss, disc_loss, sampling, config["optimizer_params"])

    return gan

if __name__ == "__main__":
    model_names = ["dcgan"]
    for model_name in model_names:
        fpath= f"./configs/{model_name}.yaml"
        config = get_config(fpath)
        gan = assembler(config)

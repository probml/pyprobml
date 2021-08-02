import importlib
import yaml
from functools import partial
from models.base_vae import VAE

def assembler(vae_name, config):
    componets = importlib.import_module(f"models.{vae_name}")
    encoder = componets.Encoder(**config["encoder_params"])
    decoder = componets.Decoder(**config["decoder_params"])
    loss = partial(componets.loss, config["loss_params"])
    vae = VAE(loss, encoder, decoder)
    return vae
    
if __name__ == "__main__":
    model_name = "vanilla_vae"
    
    with open(f"./configs/{model_name}.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    vae = assembler(model_name , config)
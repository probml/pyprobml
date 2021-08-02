import importlib
import yaml
from functools import partial
from models.base_vae import VAE

def assembler(config):
    # Check config is valid
    vae_name = config["exp_params"]["model_name"]
    assert config["encoder_params"]["latent_dim"] == config["decoder_params"]["latent_dim"]

    componets = importlib.import_module(f"models.{vae_name}")
    encoder = componets.Encoder(**config["encoder_params"])
    decoder = componets.Decoder(**config["decoder_params"])
    loss = partial(componets.loss, config["loss_params"])
    
    # Assemble my model
    vae = VAE(vae_name, loss, encoder, decoder)
    vae = VAEModule(vae, config["exp_params"]["LR"], config["encoder_params"]["latent_dim"])

    return vae
    
if __name__ == "__main__":
    model_name = "vanilla_vae"
    
    with open(f"./configs/{model_name}.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    vae = assembler(model_name , config)

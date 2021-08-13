import yaml
import importlib
from functools import partial
from models.guassian_vae import VAE
from models.two_stage_vae import Stage2VAE 
from models.vq_vae import VQVAE
from models.pixel_cnn import PixelCNN
from experiment import VAEModule, VAE2stageModule, VQVAEModule, PixelCNNModule

def get_config(fpath):
   with open(fpath, 'r') as file:
      try:
          config = yaml.safe_load(file)
      except yaml.YAMLError as exc:
          print(exc)
      return config

def  is_vq_vae(config):
    return config["exp_params"]["template"] == "vq vae"      
def is_two_stage(config):
   return config["exp_params"]["template"] == "2 stage vae"

def is_default_vae(config):
   return config["exp_params"]["template"] == "default vae"

def is_config_valid(config):
    # Check config is valid
    assert type(config["exp_params"]["model_name"]) == str 
    assert type(config["exp_params"]["template"]) == str
    assert config["encoder_params"]["latent_dim"] == config["decoder_params"]["latent_dim"]
    if is_two_stage(config):
        assert type(config["stage1_params"]["model"]) == str 
        model_name = config["stage1_params"]["model"]
        fpath= f"./configs/{model_name}.yaml"
        config_stage_one = get_config(fpath)
        is_config_valid(config_stage_one)
        assert config_stage_one["encoder_params"]["latent_dim"] == config["encoder_params"]["input_dim"]
        assert config["decoder_params"]["output_dim"] == config["encoder_params"]["input_dim"]
    elif is_vq_vae(config):
        assert config["encoder_params"]["latent_dim"] == config["vq_params"]["embedding_dim"]

def is_mode_training(mode):
    return mode=="training"

def is_mode_inference(mode):
    return mode=="inference"

def get_first_stage_vae(config):
    model_name = config["stage1_params"]["model"]
    config = get_config(f"./configs/{model_name}.yaml")
    vae = assembler(config, "inference")
    vae.load_model()
    return vae

def compose_for_inference(models):
    if len(models) == 0:
        raise "empty model list"
    elif len(models) == 1:
        return models[0]
    elif len(models) == 2:
        return VAE2stageModule(models[0], models[1])
    else:
        vae = compose_for_inference(models[:-1])
        return VAE2stageModule(vae, models[-1])

def assembler(config, mode):
    # Get model name
    is_config_valid(config)

    # Get model components 
    vae_name = config["exp_params"]["model_name"]
    componets = importlib.import_module(f"models.{vae_name}")
    encoder = componets.Encoder(**config["encoder_params"])
    decoder = componets.Decoder(**config["decoder_params"])
    loss = partial(componets.loss, config["loss_params"])
    
    # Assemble my model
    if is_default_vae(config):
        vae = VAE(vae_name, loss, encoder, decoder)
        vae = VAEModule(vae, config["exp_params"]["LR"], config["encoder_params"]["latent_dim"])
        vaes = [vae]
    elif is_two_stage(config):
        vae_first_stage = get_first_stage_vae(config)
        vae = Stage2VAE(vae_name, loss, encoder, decoder, vae_first_stage)
        vae = VAEModule(vae, config["exp_params"]["LR"], config["encoder_params"]["latent_dim"])
        vaes = [vae_first_stage , vae]
    elif is_vq_vae(config):
        vae = VQVAE(vae_name, loss, encoder, decoder, config["vq_params"])
        vae = VQVAEModule(vae, config)
        vaes = [vae]
    
    # training vs inference time model
    if is_mode_training(mode):
        vae = vaes[-1]
    elif is_mode_inference(mode):
        if is_two_stage(config):
            vae = compose_for_inference(vaes)

    return vae

if __name__ == "__main__":
    model_names = ["hinge_vae", "two_stage_vae"]
    for model_name in model_names:
        fpath= f"./configs/{model_name}.yaml"
        config = get_config(fpath)
        vae = assembler(config, "training")
        vae = assembler(config, "inference")

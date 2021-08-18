import torch
import numpy as np
import pandas as pd
from einops import rearrange
from typing import Callable
from torchvision.utils import make_grid

def get_imgs_and_attr(batch):
  imgs, attr = batch
  df = pd.DataFrame(attr.numpy(), columns=['5_o_Clock_Shadow', 'Arched_Eyebrows', 
                                            'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'])
  return imgs, df

def vector_of_interest(vae, batch, feature_of_interest="Male"):
  imgs, attr =  get_imgs_and_attr(batch)
  id = np.array(attr.index)
  get_id_of_all_absent = id[attr[feature_of_interest] == 0]
  get_id_of_all_present = id[attr[feature_of_interest] == 1]
  present = imgs[get_id_of_all_present]
  absent = imgs[get_id_of_all_absent]
  z_present = vae.det_encode(present).mean(axis=0)
  z_absent = vae.det_encode(absent).mean(axis=0)
  label_vector = z_present-z_absent
  return label_vector, present, absent

def get_interpolation(interpolation):
  """
  interpolation: can accept either string or function
  """
  if interpolation=="spherical":
    return slerp
  elif interpolation=="linear":
    return lerp 
  elif callable(interpolation):
    return interpolation

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif torch.allclose(low, high):
        return low
    omega = torch.arccos(torch.dot(low/torch.norm(low), high/torch.norm(high)))
    so = torch.sin(omega)
    return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega)/so * high

def make_imrange(arr: list):
  interpolation = torch.stack(arr)
  imgs = rearrange(make_grid(interpolation,11), 'c h w -> h w c')
  imgs = imgs.cpu().detach().numpy() if torch.cuda.is_available() else imgs.detach().numpy()
  return imgs

def get_imrange(G:Callable[[torch.tensor], torch.tensor], start:torch.tensor,
               end:torch.tensor, nums:int=8, interpolation="spherical") -> torch.tensor:
    """
    Decoder must produce a 3d vector to be appened togther to form a new grid
    """
    val = 0 
    arr2 = []
    inter = get_interpolation(interpolation)
    for val in torch.linspace(0, 1, nums):
        new_z = torch.unsqueeze(inter(val, start[0], end[0]),0)
        arr2.append(G(new_z)[0])
    return make_imrange(arr2) 
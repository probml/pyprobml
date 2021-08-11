from typing import Callable
import torch
import numpy as np
from torch import Tensor
from typing import Callable

def sampling(config: dict, G: Callable, D: Callable, z_img: Tensor):
    eta = config["eta"]
    noise_factor = config["noise_factor"]
    num_steps = config["num_steps"]

    def _velocity(z_img, D, G):
        z_img_t = z_img.clone()
        z_img_t.requires_grad_(True)
        if z_img_t.grad is not None:
            z_img_t.grad.zero_()
        d_score = D(G(z_img_t)) 
        d_score.backward(torch.ones_like(d_score).to(z_img.device))
        return z_img_t.grad.data

    def refine_samples(z_img, G, D):
        for _ in range(1, num_steps):
            v = _velocity(z_img, D, G)
            z_img = z_img.data + eta * v + np.sqrt(2*eta) * noise_factor * torch.randn_like(z_img)
        return z_img
    
    return refine_samples(z_img, G, D)
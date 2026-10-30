"""Load the official codec and move its complete model to the shared dtype."""

import torch


def load_vae(vae_pth, device="cuda", *, z_dim=16, dtype=torch.bfloat16):
    if z_dim == 16:
        from .wan21 import WanVAE as codec
    elif z_dim == 48:
        from .wan22 import Wan2_2_VAE as codec
    else:
        raise ValueError(f'Unsupported Wan latent channels: {z_dim}')
    native = codec(vae_pth=vae_pth, dtype=dtype, device='cpu')
    native.model.to(device=device, dtype=dtype)
    native.device = device
    for name, value in vars(native).items():
        if isinstance(value, torch.Tensor):
            setattr(native, name, value.to(device))
    native.scale = [value.to(device) for value in native.scale]
    return native

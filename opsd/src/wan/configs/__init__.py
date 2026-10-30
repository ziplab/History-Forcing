# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan video presets. Checkpoint config.json remains authoritative for weights."""

from copy import deepcopy

from .wan21_t2v_1_3B import t2v_1_3B as wan21_t2v_1_3B
from .wan21_t2v_14B import t2v_14B as wan21_t2v_14B
from .wan21_i2v_14B import i2v_14B as wan21_i2v_14B
from .wan22_t2v_A14B import t2v_A14B as wan22_t2v_A14B
from .wan22_i2v_A14B import i2v_A14B as wan22_i2v_A14B
from .wan22_ti2v_5B import ti2v_5B as wan22_ti2v_5B

WAN_CONFIGS = {
    'wan2.1/t2v-1.3B': wan21_t2v_1_3B,
    'wan2.1/t2v-14B': wan21_t2v_14B,
    'wan2.1/i2v-14B': wan21_i2v_14B,
    'wan2.2/t2v-A14B': wan22_t2v_A14B,
    'wan2.2/i2v-A14B': wan22_i2v_A14B,
    'wan2.2/ti2v-5B': wan22_ti2v_5B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280), '1280*720': (1280, 720),
    '480*832': (480, 832), '832*480': (832, 480),
    '704*1280': (704, 1280), '1280*704': (1280, 704),
}
MAX_AREA_CONFIGS = {name: width * height for name, (width, height) in SIZE_CONFIGS.items()}
SUPPORTED_SIZES = {
    'wan2.1/t2v-1.3B': ('480*832', '832*480'),
    'wan2.1/t2v-14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'wan2.1/i2v-14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'wan2.2/t2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'wan2.2/i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'wan2.2/ti2v-5B': ('704*1280', '1280*704'),
}


def get_config(name):
    """Return independent model and inference defaults for a released variant."""
    return deepcopy(WAN_CONFIGS[name])


def model_config(config):
    """Constructor arguments only; inference/codec settings stay outside DiT."""
    names = ('model_type', 'patch_size', 'text_len', 'in_dim', 'dim', 'ffn_dim',
             'freq_dim', 'text_dim', 'out_dim', 'num_heads', 'num_layers',
             'window_size', 'qk_norm', 'cross_attn_norm', 'eps')
    return {name: config[name] for name in names}


def codec_config(latent_channels):
    if latent_channels == 16:
        return dict(vae_checkpoint='Wan2.1_VAE.pth', vae_stride=(4, 8, 8), z_dim=16)
    if latent_channels == 48:
        return dict(vae_checkpoint='Wan2.2_VAE.pth', vae_stride=(4, 16, 16), z_dim=48)
    raise ValueError(f'Unsupported Wan latent channels: {latent_channels}')

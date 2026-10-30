# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan2.2 token-level diffusion timesteps; image latents enter the shared DiT."""

from ..layers import WanCrossAttention
from ..model import WanModelBase


class WanModel(WanModelBase):
    supported_model_types = ('t2v', 'i2v', 'ti2v')

    def cross_attention_type(self, model_type):
        return WanCrossAttention

    def init_conditioning(self):
        pass

    def image_context(self, context, clip_fea):
        if clip_fea is not None:
            raise ValueError('Wan2.2 does not use CLIP image features')
        return context

    def time_embeddings(self, t, seq_len):
        if t.ndim == 1:
            t = t[:, None].expand(-1, seq_len)
        if t.ndim != 2 or t.shape[1] != seq_len:
            raise ValueError('Wan2.2 expects sample or token timesteps')
        e, e0 = self.embed_timesteps(t)
        return e, e0.unflatten(2, (6, self.dim))

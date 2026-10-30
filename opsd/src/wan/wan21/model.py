# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan2.1 image cross-attention and sample-level diffusion timesteps."""

from functools import partial

import torch
from torch import nn

from ..attention import attention
from ..layers import WanCrossAttention, WanRMSNorm, WanSelfAttention
from ..model import WanModelBase


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, text_len=512):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.text_len = text_len
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - self.text_len
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        return self.proj(image_embeds)


class WanModel(WanModelBase):
    supported_model_types = ('t2v', 'i2v')

    def cross_attention_type(self, model_type):
        return (partial(WanI2VCrossAttention, text_len=self.text_len)
                if model_type == 'i2v' else WanCrossAttention)

    def init_conditioning(self):
        if self.model_type == 'i2v':
            self.img_emb = MLPProj(1280, self.dim)

    def image_context(self, context, clip_fea):
        if self.model_type == 'i2v':
            if clip_fea is None:
                raise ValueError('Wan2.1 I2V requires CLIP image features')
            return torch.cat((self.img_emb(clip_fea), context), dim=1)
        if clip_fea is not None:
            raise ValueError('Wan2.1 T2V does not accept CLIP image features')
        return context

    def time_embeddings(self, t, seq_len):
        if t.ndim != 1:
            raise ValueError('Native Wan2.1 expects one timestep per sample')
        e, e0 = self.embed_timesteps(t)
        return e.unsqueeze(1), e0.unflatten(1, (6, self.dim)).unsqueeze(1)

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Shared checkpoint layout and full-context forward for Wan video models."""

import math
from functools import partial

import torch
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .layers import Head, WanAttentionBlock, rope_params, embed_time


class WanModelBase(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True
    block_type = WanAttentionBlock
    head_type = Head

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 conditioning_config=None):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Native model variant supported by the Wan subclass
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        if model_type not in self.supported_model_types:
            raise ValueError(f'Unsupported {type(self).__module__} model_type: {model_type}')
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.conditioning_config = conditioning_config

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            self.block_type(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps,
                              cross_attention=self.cross_attention_type(model_type))
            for _ in range(num_layers)
        ])

        # head
        self.head = self.head_type(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        self.init_conditioning()
        self.sequence_parallel = False
        self.gradient_checkpointing = False
        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        clip_fea=None,
        **conditioning,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of noisy latent tensors, each with shape [C, F, H, W]
            t (Tensor):
                Sample timesteps [B], or token timesteps [B, seq_len] for Wan2.2
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Padded latent token sequence length
            y (List[Tensor], *optional*):
                Conditional latents aligned with x in time and space

        Returns:
            List[Tensor]:
                List of latent-space predictions with shape [C_out, F, H, W]
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = self.condition_inputs(x, y)

        # embeddings
        x = self.embed_video(x)
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        e, e0 = self.time_embeddings(t, seq_len)

        # context
        context_lens = None
        context = self.project_context(context, clip_fea)
        if self.sequence_parallel:
            from .distributed.util import get_rank, get_world_size
            world_size, rank = get_world_size(), get_rank()
            if seq_len % world_size:
                raise ValueError('seq_len must be divisible by the sequence parallel group size')
            x = x.chunk(world_size, dim=1)[rank]
            if e.shape[1] != 1:
                e = e.chunk(world_size, dim=1)[rank]
                e0 = e0.chunk(world_size, dim=1)[rank]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        kwargs.update(self.block_kwargs(x, grid_sizes, **conditioning))

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Diffusers' checkpoint callback accepts positional inputs only.
                x = self._gradient_checkpointing_func(partial(block, **kwargs), x)
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        if self.sequence_parallel:
            from .distributed.util import gather_forward
            x = gather_forward(x, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def condition_inputs(self, x, y):
        """Native latent-channel conditioning; world models may override it."""
        if y is None:
            if self.in_dim != self.out_dim:
                raise ValueError('This checkpoint requires image/video latent conditioning')
            return x
        return [torch.cat((u, v), dim=0) for u, v in zip(x, y, strict=True)]

    def embed_video(self, x):
        return [self.patch_embedding(u.unsqueeze(0)) for u in x]

    def project_context(self, context, clip_fea=None):
        context = self.text_embedding(torch.stack([
            torch.cat((u, u.new_zeros(self.text_len - u.size(0), u.size(1))))
            for u in context
        ]))
        return self.image_context(context, clip_fea)

    def embed_timesteps(self, timesteps):
        """Compute time MLPs in model precision; keep modulation in FP32."""
        embedding = embed_time(self.time_embedding, timesteps, self.freq_dim)
        projected = self.time_projection(embedding.to(self.time_projection[1].weight.dtype))
        return embedding, projected.float()

    def block_kwargs(self, tokens, grid_sizes, **conditioning):
        """Project extra world-model conditions once before the block loop."""
        if conditioning:
            raise ValueError(f'Unsupported Wan conditions: {tuple(conditioning)}')
        return {}

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Patch features with shape [B, L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Spatiotemporal patch-grid dimensions,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed latent tensors with shape [C_out, F, H, W],
                where F, H and W are the patch-grid dimensions times the
                corresponding patch sizes
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=True):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=True)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

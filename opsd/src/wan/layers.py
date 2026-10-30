# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Shared Wan DiT layers and parameter-preserving AR attention adapters."""

import math

import torch
from torch import nn

from .attention import attention


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def embed_time(embedding, timesteps, freq_dim):
    """Cast once at the time-MLP boundary; return FP32 modulation features."""
    inputs = sinusoidal_embedding_1d(freq_dim, timesteps.flatten()).to(embedding[0].weight.dtype)
    return embedding(inputs).float().unflatten(0, timesteps.shape)


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return torch.nn.functional.layer_norm(
            x.float(), self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(), self.eps,
        ).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def project_qkv(self, x):
        batch, frames = x.shape[:2]
        shape = (batch, frames, self.num_heads, self.head_dim)
        return (self.norm_q(self.q(x)).view(shape),
                self.norm_k(self.k(x)).view(shape), self.v(x).view(shape))

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.project_qkv(x)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    self_attention_type = WanSelfAttention

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 cross_attention=WanCrossAttention):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = self.self_attention_type(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = cross_attention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        x = x + y * e[2].squeeze(2)

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        return x + y * e[5].squeeze(2)


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        return self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))


class Attention(nn.Module):
    """Reuse native projections without copying weights or adding key prefixes."""

    project_qkv = WanSelfAttention.project_qkv

    def __init__(self, native):
        super().__init__()
        self.dim = native.dim
        self.num_heads = native.num_heads
        self.head_dim = native.head_dim
        for name, child in native.named_children():
            self.add_module(name, child)


class CachedCrossAttention(Attention):
    """Cache native text/image KVs and route two text prompts during SFT."""

    def __init__(self, native):
        super().__init__(native)
        self.text_len = getattr(native, "text_len", None)

    def forward(
        self,
        x,
        context,
        context_lens,
        crossattn_cache=None,
        secondary_context=None,
        secondary_context_lens=None,
        context_mask=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        image_output = None
        if self.text_len is not None:
            image_context, context = (
                context[:, :-self.text_len], context[:, -self.text_len:]
            )
            if not image_context.shape[1]:
                raise ValueError("Wan2.1 I2V requires projected image context")
            if crossattn_cache is not None and crossattn_cache["is_init"]:
                image_k, image_v = crossattn_cache["image_k"], crossattn_cache["image_v"]
            else:
                image_k = self.norm_k_img(self.k_img(image_context)).view(b, -1, n, d)
                image_v = self.v_img(image_context).view(b, -1, n, d)
                if crossattn_cache is not None:
                    crossattn_cache["image_k"], crossattn_cache["image_v"] = image_k, image_v
            image_output = attention(q, image_k, image_v)
            if secondary_context is not None:
                secondary_context = secondary_context[:, -self.text_len:]

        if crossattn_cache is None or not crossattn_cache["is_init"]:
            if crossattn_cache is not None:
                crossattn_cache["is_init"] = True
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            if crossattn_cache is not None:
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
        else:
            k = crossattn_cache["k"]
            v = crossattn_cache["v"]

        # SFT may route each visual frame to one of two text prompts.  The
        # projections are shared: compute Q once, attend to both text K/V
        # sets, then select before the shared output projection.
        primary = attention(q, k, v, k_lens=context_lens)
        if secondary_context is None:
            if secondary_context_lens is not None or context_mask is not None:
                raise ValueError(
                    "A cross-attention context mask requires a secondary context"
                )
            x = primary
        else:
            if crossattn_cache is not None:
                raise ValueError(
                    "Dual-prompt cross-attention is training-only and cannot "
                    "reuse a single inference text cache"
                )
            if context_mask is None or context_mask.shape != x.shape[:2] or (
                context_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "Cross-attention context mask must be boolean [B,L]"
                )
            secondary_k = self.norm_k(self.k(secondary_context)).view(
                b, -1, n, d
            )
            secondary_v = self.v(secondary_context).view(b, -1, n, d)
            secondary = attention(
                q,
                secondary_k,
                secondary_v,
                k_lens=secondary_context_lens,
            )
            x = torch.where(
                context_mask[:, :, None, None], secondary, primary
            )

        # output
        x = x.flatten(2)
        if image_output is not None:
            x = x + image_output.flatten(2)
        x = self.o(x)
        return x

"""Shared block-causal Wan backbone for training and cached inference."""

import math
from functools import partial

import torch
from wan.conditioning import TemporalLayout
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from wan.kv_cache.causal import read_attention

from .attention import attention
from .layers import Attention, CachedCrossAttention
from .model import WanModelBase
from . import model_class


_flex_attention = torch.compile(flex_attention, dynamic=False, mode="default")
_FLEX_ATTENTION_KERNEL_OPTIONS = {
    "fwd_BLOCK_M": 64,
    "fwd_BLOCK_N": 64,
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 32,
    "bwd_BLOCK_N2": 32,
}


def causal_rope_apply(x, grid_sizes, freqs, start_frame):
    """Apply Wan RoPE to a contiguous latent-frame block."""
    heads, complex_dim = x.shape[2], x.shape[3] // 2
    frequency_parts = freqs.split(
        [
            complex_dim - 2 * (complex_dim // 3),
            complex_dim // 3,
            complex_dim // 3,
        ],
        dim=1,
    )
    output = []
    for sample, (frames, height, width) in zip(
        x, grid_sizes.tolist(), strict=True
    ):
        sequence_length = frames * height * width
        values = torch.view_as_complex(
            sample[:sequence_length].to(torch.float64).reshape(
                sequence_length, heads, -1, 2
            )
        )
        frame_frequencies = frequency_parts[0][
            start_frame:start_frame + frames
        ]
        if len(frame_frequencies) != frames:
            raise ValueError("Causal RoPE frame index exceeds its table")
        sample_frequencies = torch.cat(
            [
                frame_frequencies[:, None, None].expand(
                    frames, height, width, -1
                ),
                frequency_parts[1][:height][None, :, None].expand(
                    frames, height, width, -1
                ),
                frequency_parts[2][:width][None, None, :].expand(
                    frames, height, width, -1
                ),
            ],
            dim=-1,
        ).reshape(sequence_length, 1, -1)
        values = torch.view_as_real(
            values * sample_frequencies
        ).flatten(2)
        output.append(torch.cat((values, sample[sequence_length:])))
    return torch.stack(output).type_as(x)


def indexed_rope_apply(x, positions, grid_sizes, freqs):
    """Apply Wan RoPE at retained token coordinates, including sparse history."""
    d = x.shape[-1] // 2
    temporal, height, width = freqs.split((d - 2 * (d // 3), d // 3, d // 3), -1)
    grid_sizes = grid_sizes.to(device=positions.device)
    w = grid_sizes[:, 2:3]
    tokens = grid_sizes[:, 1:2] * w
    phases = torch.cat((temporal[positions // tokens], height[(positions % tokens) // w],
                        width[positions % w]), -1)
    values = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(values * phases[:, :, None]).flatten(-2).type_as(x)


class CausalWanSelfAttention(Attention):
    """Wan self-attention with teacher-forcing and inference-cache paths."""

    def __init__(self, native, local_attn_size=-1, sink_size=0):
        super().__init__(native)
        if type(local_attn_size) is not int or (
            local_attn_size != -1 and local_attn_size <= 0
        ):
            raise ValueError("local_attn_size must be -1 or a positive integer")
        if type(sink_size) is not int or sink_size < 0:
            raise ValueError("sink_size must be a non-negative integer")
        if local_attn_size == -1 and sink_size:
            raise ValueError("A bounded attention sink requires a local window")
        if local_attn_size != -1 and sink_size >= local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if self.head_dim < 16:
            raise ValueError("Causal Wan attention head dimension must be at least 16")
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

    @staticmethod
    def _padded_flex_attention(query, key, value, block_mask):
        sequence = query.shape[1]
        padding = (-sequence) % 128
        if padding:
            shape = (query.shape[0], padding, query.shape[2], query.shape[3])
            query = torch.cat((query, query.new_zeros(shape)), dim=1)
            key = torch.cat((key, key.new_zeros(shape)), dim=1)
            value = torch.cat((value, value.new_zeros(shape)), dim=1)
        output = _flex_attention(
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
            block_mask=block_mask,
            kernel_options=_FLEX_ATTENTION_KERNEL_OPTIONS,
        ).transpose(1, 2)
        return output[:, :sequence]

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask=None,
        kv_cache=None,
        current_start=0,
        commit_cache=True,
        position_start=None,
        extra_sink_size=0,
        **conditioning,
    ):
        sequence = x.shape[1]
        query, key, value = self.project_qkv(x)

        if kv_cache is None:
            if block_mask is None:
                raise ValueError("Teacher forcing requires a causal block mask")
            target_sequence = int(seq_lens[0])
            prefix_sequence = sequence - 2 * target_sequence
            tokens_per_frame = math.prod(grid_sizes[0, 1:].tolist())
            if prefix_sequence < 0 or prefix_sequence % tokens_per_frame:
                raise ValueError(
                    "Teacher-forcing sequence has an invalid clean prefix"
                )
            prefix_frames = prefix_sequence // tokens_per_frame
            temporal_prefix_frames = prefix_frames
            query_prefix = query[:, :prefix_sequence]
            key_prefix = key[:, :prefix_sequence]
            query_clean = query[
                :, prefix_sequence:prefix_sequence + target_sequence
            ]
            key_clean = key[:, prefix_sequence:prefix_sequence + target_sequence]
            query_noisy = query[:, prefix_sequence + target_sequence:]
            key_noisy = key[:, prefix_sequence + target_sequence:]
            prefix_grid = grid_sizes.clone()
            prefix_grid[:, 0] = prefix_frames
            roped_query_prefix = (
                causal_rope_apply(query_prefix, prefix_grid, freqs, 0)
                if prefix_sequence else query_prefix
            )
            roped_key_prefix = (
                causal_rope_apply(key_prefix, prefix_grid, freqs, 0)
                if prefix_sequence else key_prefix
            )
            roped_query = torch.cat(
                (
                    roped_query_prefix,
                    causal_rope_apply(
                        query_clean,
                        grid_sizes,
                        freqs,
                        temporal_prefix_frames,
                    ),
                    causal_rope_apply(
                        query_noisy,
                        grid_sizes,
                        freqs,
                        temporal_prefix_frames,
                    ),
                ),
                dim=1,
            ).type_as(value)
            roped_key = torch.cat(
                (
                    roped_key_prefix,
                    causal_rope_apply(
                        key_clean,
                        grid_sizes,
                        freqs,
                        temporal_prefix_frames,
                    ),
                    causal_rope_apply(
                        key_noisy,
                        grid_sizes,
                        freqs,
                        temporal_prefix_frames,
                    ),
                ),
                dim=1,
            ).type_as(value)
            output = self._padded_flex_attention(
                roped_query, roped_key, value, block_mask
            )
        else:
            tokens_per_frame = math.prod(grid_sizes[0, 1:].tolist())
            if current_start % tokens_per_frame:
                raise ValueError("KV cache start must lie on a latent-frame boundary")
            if position_start is None:
                position_start = current_start
            if position_start % tokens_per_frame:
                raise ValueError(
                    "RoPE start must lie on a latent-frame boundary"
                )
            if type(extra_sink_size) is not int or extra_sink_size < 0:
                raise ValueError("Extra attention sink size must be non-negative")
            start_frame = position_start // tokens_per_frame
            sink_tokens = kv_cache.get('sink_tokens', (self.sink_size + extra_sink_size) * tokens_per_frame)
            roped_query = causal_rope_apply(
                query, grid_sizes, freqs, start_frame
            ).type_as(value)
            roped_key = causal_rope_apply(
                key, grid_sizes, freqs, start_frame
            ).type_as(value)
            attention_key, attention_value, key_lengths = read_attention(
                kv_cache, roped_key, value, current_start, sink_tokens, tokens_per_frame,
                commit=commit_cache, freqs=freqs)
            output = attention(
                roped_query,
                attention_key,
                attention_value,
                k_lens=key_lengths,
            )

        output = self.o(output.flatten(2))
        conditioned_output = self.condition_attention(
            query, key, value,
            block_mask=block_mask, use_cache=kv_cache is not None,
            current_start=current_start, tokens_per_frame=tokens_per_frame,
            extra_sink_size=extra_sink_size, commit_cache=commit_cache,
            **conditioning,
        )
        return output if conditioned_output is None else output + conditioned_output

    def condition_attention(
        self, query, key, value, *, block_mask, use_cache,
        current_start, tokens_per_frame, extra_sink_size, commit_cache, **conditioning,
    ):
        if conditioning:
            raise ValueError(f"Unsupported Wan attention inputs: {tuple(conditioning)}")
        return None


class CausalWanAttentionBlock(nn.Module):
    attention_type = CausalWanSelfAttention
    cross_attention_type = CachedCrossAttention

    def __init__(self, native, local_attn_size=-1, sink_size=0):
        super().__init__()
        self.norm1 = native.norm1
        self.self_attn = self.attention_type(native.self_attn, local_attn_size, sink_size)
        self.norm3 = native.norm3
        self.cross_attn = self.cross_attention_type(native.cross_attn)
        self.norm2 = native.norm2
        self.ffn = native.ffn
        self.modulation = native.modulation

    def inject_camera(self, x, camera_features):
        if camera_features is not None:
            raise ValueError("This Wan block does not accept camera features")
        return x

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        commit_cache=True,
        position_start=None,
        extra_sink_size=0,
        secondary_context=None,
        context_mask=None,
        camera_features=None,
        **attention_inputs,
    ):
        frames = e.shape[1]
        if x.shape[1] % frames:
            raise ValueError("Wan tokens do not divide evenly into latent frames")
        tokens_per_frame = x.shape[1] // frames
        # Official Wan modulation and residual arithmetic, with an AR frame axis.
        modulation = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        normalized = self.norm1(x).float().unflatten(
            1, (frames, tokens_per_frame)
        )
        attention_input = (
            normalized * (1 + modulation[1]) + modulation[0]
        ).flatten(1, 2)
        residual = self.self_attn(
            attention_input,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            commit_cache=commit_cache,
            position_start=position_start,
            extra_sink_size=extra_sink_size,
            **attention_inputs,
        )
        x = x + (residual.unflatten(1, (frames, tokens_per_frame)) * modulation[2]).flatten(1, 2)
        x = self.inject_camera(x, camera_features)
        x = x + self.cross_attn(
            self.norm3(x),
            context,
            context_lens,
            crossattn_cache=crossattn_cache,
            secondary_context=secondary_context,
            secondary_context_lens=None,
            context_mask=context_mask,
        )
        normalized = self.norm2(x).float().unflatten(
            1, (frames, tokens_per_frame)
        )
        residual = self.ffn(
            (normalized * (1 + modulation[4]) + modulation[3]).flatten(1, 2)
        )
        x = x + (residual.unflatten(1, (frames, tokens_per_frame)) * modulation[5]).flatten(1, 2)
        return x


class CausalHead(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.patch_size = native.patch_size
        self.norm = native.norm
        self.head = native.head
        self.modulation = native.modulation
        self._native_forward = type(native).forward

    def forward(self, x, e):
        batch, frames = e.shape[:2]
        tokens_per_frame = x.shape[1] // frames
        x = x.reshape(batch * frames, tokens_per_frame, -1)
        e = e.flatten(0, 1)
        output = self._native_forward(self, x, e)
        return output.unflatten(0, (batch, frames)).flatten(1, 2)


class CausalWanModel(ModelMixin, ConfigMixin):
    """Shared Wan AR backbone with backend-owned camera conditioning."""

    _no_split_modules = ["CausalWanAttentionBlock"]
    _supports_gradient_checkpointing = True
    block_type = CausalWanAttentionBlock
    head_type = CausalHead
    supported_model_types = ("t2v", "i2v", "ti2v")
    temporal_layout = TemporalLayout(4)
    base = "wan2.1"
    cache_streams = ('kv_cache',)

    def condition_kwargs(self, inputs, batch, frames, *, prefix=False):
        """Map prepared model inputs to native forward names, before projection.

        Subclasses validate or rename their controls here, including how
        shared static conditions participate in clean prefix projection.
        """
        return {f'prefix_{name}' if prefix else name: value for name, value in inputs.items()}

    @property
    def num_frame_per_block(self):
        return self.temporal_layout.block_size

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
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
        local_attn_size=20,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        control_type=None,
        base=None,
    ):
        super().__init__()
        if model_type not in self.supported_model_types:
            raise ValueError(f"{type(self).__name__} does not support model_type={model_type}")
        self.base = base or type(self).base
        self.register_to_config(base=self.base)
        if dim % num_heads or (dim // num_heads) % 2:
            raise ValueError("Wan attention head dimensions are invalid")
        native = self.native_model_class()(
            model_type=model_type, patch_size=patch_size, text_len=text_len,
            in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim,
            text_dim=text_dim, out_dim=out_dim, num_heads=num_heads,
            num_layers=num_layers, qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm, eps=eps,
        )
        for name in (
            "model_type", "patch_size", "text_len", "in_dim", "dim", "ffn_dim", "freq_dim",
            "text_dim", "out_dim", "num_heads", "num_layers", "freqs",
        ):
            setattr(self, name, getattr(native, name))
        # Adopt native and backend-added embeddings without extra checkpoint prefixes.
        for name, child in native.named_children():
            if name not in ("blocks", "head"):
                self.add_module(name, child)
        for name, parameter in native.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in native.named_buffers(recurse=False):
            self.register_buffer(name, buffer, persistent=name not in native._non_persistent_buffers_set)
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.blocks = nn.ModuleList([
            self.block_type(block, local_attn_size, sink_size) for block in native.blocks
        ])
        self.head = self.head_type(native.head)
        self._init_conditioning(control_type)
        self.gradient_checkpointing = False
        self._teacher_masks = {}

    def native_model_class(self):
        """Override when a world model changes the native checkpoint structure."""
        return model_class(self.base)

    def _init_conditioning(self, control_type):
        if control_type is not None:
            raise ValueError("This Wan model does not accept an external control branch")

    def _project_camera(self, camera_embedding):
        if camera_embedding is not None:
            raise ValueError("This Wan model does not accept camera embeddings")
        return None

    @staticmethod
    def _teacher_forcing_mask(
        device,
        frames,
        tokens_per_frame,
        frame_blocks,
        prefix_blocks,
        prefix_frames,
        local_attn_size,
        sink_size,
        extra_sink_size,
    ):
        prefix_length = prefix_frames * tokens_per_frame
        half_length = frames * tokens_per_frame
        total_length = prefix_length + 2 * half_length
        padded_length = math.ceil(total_length / 128) * 128
        context_ends = torch.zeros(
            padded_length, device=device, dtype=torch.long
        )
        context_starts = torch.zeros_like(context_ends)
        noise_context_starts = torch.zeros_like(context_ends)
        noise_context_ends = torch.zeros_like(context_ends)
        noise_starts = torch.zeros_like(context_ends)
        noise_ends = torch.zeros_like(context_ends)
        local_tokens = (
            None
            if local_attn_size == -1
            else local_attn_size * tokens_per_frame
        )
        temporal_sink_tokens = sink_size * tokens_per_frame
        sink_tokens = extra_sink_size * tokens_per_frame + temporal_sink_tokens
        recent_capacity = None if local_tokens is None else max(0, local_tokens - temporal_sink_tokens)
        frame_blocks = tuple(frame_blocks)
        # Clean prefix and target share the same block-causal history window.
        # Noisy targets below see only earlier clean blocks and their own noise.
        clean_frame_start = 0
        for frame_count in (*prefix_blocks, *frame_blocks):
            start = clean_frame_start * tokens_per_frame
            stop = start + frame_count * tokens_per_frame
            clean_frame_start += frame_count
            context_ends[start:stop] = stop
            if recent_capacity is not None:
                context_starts[start:stop] = max(
                    sink_tokens, stop - recent_capacity
                )
        frame_start = 0
        for frame_count in frame_blocks:
            start = prefix_length + frame_start * tokens_per_frame
            stop = start + frame_count * tokens_per_frame
            frame_start += frame_count
            noisy_start = half_length + start
            noisy_stop = half_length + stop
            noise_context_ends[noisy_start:noisy_stop] = start
            if local_tokens is not None:
                prior_capacity = max(
                    0,
                    local_tokens - temporal_sink_tokens - (stop - start),
                )
                noise_context_starts[noisy_start:noisy_stop] = max(
                    sink_tokens, start - prior_capacity
                )
            noise_starts[noisy_start:noisy_stop] = noisy_start
            noise_ends[noisy_start:noisy_stop] = noisy_stop

        def mask(batch, head, query, key):
            noisy_start = prefix_length + half_length
            clean = (
                (query < noisy_start)
                & (key < context_ends[query])
                & (
                    (key < sink_tokens)
                    | (key >= context_starts[query])
                )
            )
            noisy = (query >= noisy_start) & (
                (
                    (key >= noise_starts[query])
                    & (key < noise_ends[query])
                )
                | (
                    (key < noise_context_ends[query])
                    & (
                        (key < sink_tokens)
                        | (key >= noise_context_starts[query])
                    )
                )
            )
            return (query == key) | clean | noisy

        return create_block_mask(
            mask,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            device=device,
            _compile=False,
        )

    def _get_teacher_mask(
        self, device, frames, tokens_per_frame, prefix_frames
    ):
        key = (
            device.type,
            device.index,
            frames,
            tokens_per_frame,
            self.temporal_layout,
            prefix_frames,
            self.local_attn_size,
            self.sink_size,
        )
        if key not in self._teacher_masks:
            frame_blocks = self.temporal_layout.target_blocks(frames, prefix_frames=prefix_frames)
            prefix_blocks = self.temporal_layout.prefix_blocks(prefix_frames)
            self._teacher_masks[key] = self._teacher_forcing_mask(
                device,
                frames,
                tokens_per_frame,
                frame_blocks,
                prefix_blocks,
                prefix_frames,
                self.local_attn_size,
                self.sink_size,
                self.temporal_layout.extra_sink_frames(prefix_frames),
            )
        return self._teacher_masks[key]

    def _project_context(self, context, dtype, clip_fea=None):
        if torch.is_tensor(context):
            if context.ndim != 3:
                raise ValueError(
                    "Text context tensor must have shape [B,L,D]"
                )
            context = context.unbind(0)
        else:
            context = tuple(context)
        if not context:
            raise ValueError("Text context cannot be empty")
        if any(
            item.ndim != 2
            or item.shape[0] > self.text_len
            or item.shape[1] != self.text_dim
            for item in context
        ):
            raise ValueError(
                f"Each text context must have shape [L,{self.text_dim}] "
                f"with L <= {self.text_len}"
            )
        context = tuple(item.to(dtype=dtype) for item in context)
        projected = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        (
                            item,
                            item.new_zeros(
                                self.text_len - item.shape[0], item.shape[1]
                            ),
                        )
                    )
                    for item in context
                ]
            )
        )
        if hasattr(self, "img_emb"):
            if clip_fea is None:
                raise ValueError("Wan2.1 I2V requires CLIP image features")
            projected = torch.cat((self.img_emb(clip_fea), projected), dim=1)
        elif clip_fea is not None:
            raise ValueError("This Wan checkpoint does not use CLIP image features")
        return projected

    def _time_embeddings(self, timestep):
        embedding, projected = self.embed_timesteps(timestep)
        return embedding, projected.unflatten(2, (6, self.dim))

    embed_timesteps = WanModelBase.embed_timesteps

    def block_kwargs(self, tokens, grid_sizes, *, prefix_frames=0, teacher_forcing=False,
                     camera_embedding=None, prefix_camera_embedding=None):
        """Project conditions once for the native block-token layout.

        Training tokens are [prefix, clean target, noisy target]; inference
        tokens contain only the current block. Subclasses may supply any named
        inputs consumed by their block, without replacing either forward path.
        """
        if prefix_camera_embedding is not None and not prefix_frames:
            raise ValueError("Prefix camera embeddings require a clean prefix")
        features = self._project_camera(camera_embedding)
        if features is None:
            if prefix_camera_embedding is not None:
                raise ValueError("Prefix camera embeddings require target camera embeddings")
            return {}
        target_shape = (tokens.shape[0], math.prod(grid_sizes[0].tolist()), self.dim)
        if features.shape != target_shape:
            raise ValueError('Camera features must match the target patch grid')
        if teacher_forcing:
            prefix = (self._project_camera(prefix_camera_embedding)
                      if prefix_frames else features[:, :0])
            prefix_shape = (tokens.shape[0], prefix_frames * math.prod(grid_sizes[0, 1:].tolist()), self.dim)
            if prefix is None or prefix.shape != prefix_shape:
                raise ValueError('Prefix camera features must match the prefix patch grid')
            features = torch.cat((prefix, features, features), dim=1)
        if features.shape != tokens.shape:
            raise ValueError('Camera features must align with all block tokens')
        return {'camera_features': features}

    def _run_blocks(
        self, x, e, seq_lens, grid_sizes, context, block_mask, block_inputs, *,
        caches=None, crossattn_cache=None, current_start=0, commit_cache=True,
        position_start=None, extra_sink_size=0, secondary_context=None, context_mask=None,
    ):
        if (secondary_context is None) != (context_mask is None):
            raise ValueError("Secondary text context and its visual-token mask must be provided together")
        if context_mask is not None and (
            context_mask.shape != x.shape[:2] or context_mask.dtype != torch.bool
        ):
            raise ValueError("Text context mask must be boolean [B,L]")
        kwargs = {
            "e": e, "seq_lens": seq_lens, "grid_sizes": grid_sizes, "freqs": self.freqs,
            "context": context, "context_lens": None, "block_mask": block_mask,
            "current_start": current_start, "commit_cache": commit_cache,
            "position_start": position_start, "extra_sink_size": extra_sink_size,
            "secondary_context": secondary_context, "context_mask": context_mask,
            **block_inputs,
        }
        for index, block in enumerate(self.blocks):
            for name in self.cache_streams:
                layers = None if caches is None else caches.get(name)
                kwargs[name] = None if layers is None else layers[index]
            kwargs["crossattn_cache"] = None if crossattn_cache is None else crossattn_cache[index]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(partial(block, **kwargs), x)
            else:
                x = block(x, **kwargs)
        return x

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x,
        aug_t=None,
        clip_fea=None,
        image_condition=None,
        patch_embeddings=None,
        clean_patch_embeddings=None,
        prefix_embeddings=None,
        secondary_context=None,
        prompt_mask=None,
        **conditioning,
    ):
        batch, _, frames, _, _ = x.shape
        if clean_x is None or clean_x.shape != x.shape:
            raise ValueError("AR training requires clean_x matching noisy x")
        if image_condition is not None:
            x = torch.cat((x, image_condition), dim=1)
            clean_x = torch.cat((clean_x, image_condition), dim=1)
        if t.shape != (batch, frames):
            raise ValueError("AR timesteps must have shape [B,F]")
        if aug_t is None:
            aug_t = torch.zeros_like(t)
        if aug_t.shape != t.shape:
            raise ValueError("Clean-context timesteps must have shape [B,F]")
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        noisy_embedding = (
            self.patch_embedding(x)
            if patch_embeddings is None else patch_embeddings
        )
        expected_embedding = (
            batch,
            self.dim,
            frames // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )
        if noisy_embedding.shape != expected_embedding:
            raise ValueError(
                "patch_embeddings do not match the noisy latent patch grid"
            )
        clean_embedding = (
            self.patch_embedding(clean_x)
            if clean_patch_embeddings is None else clean_patch_embeddings
        )
        if clean_embedding.shape != noisy_embedding.shape:
            raise ValueError("Clean and noisy Wan patch grids differ")
        grid_sizes = torch.as_tensor(
            [noisy_embedding.shape[2:]] * batch,
            dtype=torch.long,
        )
        noisy_tokens = noisy_embedding.flatten(2).transpose(1, 2)
        clean_tokens = clean_embedding.flatten(2).transpose(1, 2)
        sequence = noisy_tokens.shape[1]
        if seq_len != sequence:
            raise ValueError(f"seq_len={seq_len} does not match {sequence}")
        if prefix_embeddings is None:
            prefix_frames = 0
            prefix_tokens = clean_tokens[:, :0]
        else:
            if (
                prefix_embeddings.ndim != 5
                or prefix_embeddings.shape[0] != batch
                or prefix_embeddings.shape[1] != self.dim
                or prefix_embeddings.shape[3:] != noisy_embedding.shape[3:]
            ):
                raise ValueError(
                    "prefix_embeddings must match the target spatial patch grid"
                )
            prefix_frames = prefix_embeddings.shape[2]
            if prefix_frames == 0:
                raise ValueError("Condition embeddings must contain at least one latent frame")
            self.temporal_layout.prefix_blocks(prefix_frames)
            prefix_tokens = prefix_embeddings.flatten(2).transpose(1, 2)
        tokens = torch.cat((prefix_tokens, clean_tokens, noisy_tokens), dim=1)
        seq_lens = torch.full((batch,), sequence, dtype=torch.long)
        noisy_head, noisy_modulation = self._time_embeddings(t)
        _, clean_modulation = self._time_embeddings(aug_t)
        prefix_timestep = t.new_zeros(batch, prefix_frames)
        _, prefix_modulation = self._time_embeddings(prefix_timestep)
        modulation = torch.cat(
            (prefix_modulation, clean_modulation, noisy_modulation), dim=1
        )
        projected_context = self._project_context(context, tokens.dtype, clip_fea)
        if (secondary_context is None) != (prompt_mask is None):
            raise ValueError(
                "Secondary context and target-frame prompt mask must be "
                "provided together"
            )
        projected_secondary_context = None
        context_mask = None
        if secondary_context is not None:
            if prompt_mask.shape != (batch, frames) or (
                prompt_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "Target-frame prompt mask must be boolean [B,F]"
                )
            projected_secondary_context = self._project_context(
                secondary_context, tokens.dtype, clip_fea
            )
            tokens_per_frame = sequence // frames
            # A condition precedes the target on the same timeline, so it
            # inherits the route selected for the first generated frame.
            prefix_prompt_mask = (
                prompt_mask[:, :1].expand(-1, prefix_frames)
                if prefix_frames else prompt_mask[:, :0]
            )
            frame_prompt_mask = torch.cat(
                (prefix_prompt_mask, prompt_mask, prompt_mask), dim=1
            )
            context_mask = frame_prompt_mask[:, :, None].expand(
                -1, -1, tokens_per_frame
            ).reshape(batch, -1)
            if context_mask.shape != tokens.shape[:2]:
                raise RuntimeError(
                    "Expanded prompt mask does not align with visual tokens"
                )
        block_mask = self._get_teacher_mask(
            device, frames, sequence // frames, prefix_frames)
        block_inputs = self.block_kwargs(tokens, grid_sizes, prefix_frames=prefix_frames,
                                         teacher_forcing=True, **conditioning)
        tokens = self._run_blocks(
            tokens, modulation, seq_lens, grid_sizes, projected_context, block_mask, block_inputs,
            secondary_context=projected_secondary_context, context_mask=context_mask,
        )[:, prefix_tokens.shape[1] + sequence:]
        output = self.head(tokens, noisy_head.unsqueeze(2))
        return torch.stack(self.unpatchify(output, grid_sizes))

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        kv_cache,
        crossattn_cache=None,
        current_start=0,
        patch_embeddings=None,
        commit_cache=True,
        position_start=None,
        extra_sink_size=0,
        secondary_context=None,
        prompt_mask=None,
        image_condition=None,
        clip_fea=None,
        **conditioning,
    ):
        if secondary_context is not None or prompt_mask is not None:
            raise ValueError(
                "Dual-prompt routing is supported only by parallel SFT; "
                "autoregressive inference switches prompts with distinct caches"
            )
        batch, _, frames, _, _ = x.shape
        if t.shape != (batch, frames):
            raise ValueError("AR inference timesteps must have shape [B,F]")
        if image_condition is not None:
            x = torch.cat((x, image_condition), dim=1)
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        embedding = (
            self.patch_embedding(x)
            if patch_embeddings is None else patch_embeddings
        )
        expected_embedding = (
            batch,
            self.dim,
            frames // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )
        if embedding.shape != expected_embedding:
            raise ValueError(
                "patch_embeddings do not match the noisy latent patch grid"
            )
        grid_sizes = torch.as_tensor(
            [embedding.shape[2:]] * batch,
            dtype=torch.long,
        )
        tokens = embedding.flatten(2).transpose(1, 2)
        if seq_len != tokens.shape[1]:
            raise ValueError(
                f"seq_len={seq_len} does not match block length "
                f"{tokens.shape[1]}"
            )
        seq_lens = torch.full(
            (batch,), tokens.shape[1], dtype=torch.long
        )
        head_embedding, modulation = self._time_embeddings(t)
        projected_context = self._project_context(context, tokens.dtype, clip_fea)
        caches = {name: conditioning.pop(name, None) for name in self.cache_streams if name != 'kv_cache'}
        caches['kv_cache'] = kv_cache
        block_inputs = self.block_kwargs(tokens, grid_sizes, **conditioning)
        tokens = self._run_blocks(
            tokens, modulation, seq_lens, grid_sizes, projected_context, None, block_inputs,
            caches=caches, crossattn_cache=crossattn_cache, current_start=current_start,
            commit_cache=commit_cache, position_start=position_start, extra_sink_size=extra_sink_size,
        )
        output = self.head(tokens, head_embedding.unsqueeze(2))
        output = torch.stack(self.unpatchify(output, grid_sizes))
        return output

    def forward(self, *args, **kwargs):
        from utils import model_autocast
        with model_autocast(self):
            if kwargs.get("kv_cache") is not None:
                return self._forward_inference(*args, **kwargs).float()
            return self._forward_train(*args, **kwargs).float()

    unpatchify = WanModelBase.unpatchify

# Adapted from AMAP CV LAB ABot-World, Apache-2.0.
"""ABot-World-0: Wan2.2 action conditioning, AR inference and training forwards.

Input tensors follow the shared target/condition contract in src/dataset.py.
This module contains no dataset readers, collation or media file handling.
"""

import json
import math
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from diffusers.configuration_utils import register_to_config

from .wan_ar import WanAR
from model import WanModelWrapper
from scheduler import make_flow_match_scheduler, parse_sampling_steps
from wan.attention import attention
from wan.causal import causal_rope_apply, CausalWanSelfAttention
from wan.conditioning import TemporalInputs
from wan.layers import Head, WanAttentionBlock, WanSelfAttention
from wan.wan22.model import WanModel


_REF_FREQS_CACHE = {}


def freqs_at(indices, dim, theta=10000, device='cpu'):
    assert dim % 2 == 0
    t = torch.tensor(indices, dtype=torch.float64, device=device)
    freqs = torch.outer(
        t,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=device, dtype=torch.float64).div(dim))
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply_with_refimg(x, freqs, num_heads):
    if x.dim() == 3:
        b, s, _ = x.shape
        x = x.view(b, s, num_heads, -1)
    # Apply RoPE with complex64 arithmetic.
    x_out = torch.view_as_complex(x.to(torch.float32).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2
    ))
    freqs = freqs.to(device=x.device)
    if freqs.dtype != torch.complex64:
        freqs = freqs.to(torch.complex64)
    x_out = torch.view_as_real(x_out * freqs).flatten(3)
    return x_out.to(x.dtype)


def _build_ref_freqs(freqs, num_slots, tokens_per_slot, ref_grid, device):
    patch_t, patch_h, patch_w = [int(v) for v in ref_grid]
    freq_dim = int(freqs.shape[1])
    cache_key = (
        int(num_slots), int(tokens_per_slot), patch_t, patch_h, patch_w,
        freq_dim, str(device),
    )
    cached = _REF_FREQS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    f_band = freq_dim - 2 * (freq_dim // 3)
    h_band = freq_dim // 3
    w_band = freq_dim // 3
    temporal_step = max(int(tokens_per_slot), 256)
    neg_temporal = [-(int(num_slots) - i) * temporal_step for i in range(int(num_slots))]
    t_freqs = freqs_at(neg_temporal, 2 * f_band, device=device)
    freqs_split = freqs.split([f_band, h_band, w_band], dim=1)
    h_freqs = freqs_split[1][:patch_h].to(device)
    w_freqs = freqs_split[2][:patch_w].to(device)
    ref_freqs = torch.cat([
        t_freqs[:, None, None, None, :].expand(num_slots, patch_t, patch_h, patch_w, f_band),
        h_freqs[None, None, :, None, :].expand(num_slots, patch_t, patch_h, patch_w, h_band),
        w_freqs[None, None, None, :, :].expand(num_slots, patch_t, patch_h, patch_w, w_band),
    ], dim=-1).reshape(int(num_slots) * int(tokens_per_slot), 1, -1).to(torch.complex64)
    _REF_FREQS_CACHE[cache_key] = ref_freqs
    return ref_freqs


def allocate(model, batch, frames, height, width, device, dtype, reference_tokens=1280,
             *, capacity_frames=None, sink_frames=None):
    from wan.kv_cache.causal import allocate_cache, cache_capacity_frames, resolve_sink_frames

    tpf = height // model.patch_size[1] * (width // model.patch_size[2])
    sink = resolve_sink_frames(sink_frames, 0)
    capacity = (cache_capacity_frames(frames, model.local_attn_size, sink,
                                      block_frames=model.num_frame_per_block)
                if capacity_frames is None else capacity_frames)
    capacity = reference_tokens + capacity * tpf
    return allocate_cache(model, batch, capacity, device, dtype, raw_keys=True, sink_tokens=sink * tpf)


ROPE_REBASE_FRAMES = 256


def cache_attention(q, k, v, cache, *, start, frames, tokens_per_frame,
                    reference_tokens, rope_video, rope_reference, attend, commit, freqs):
    """Construct the current attention view; only clean commits mutate history.

    Official inference overwrites the noisy block at each step. Reading the
    committed past plus the current differentiable K/V gives the same view
    without detaching student keys or mutating checkpoint inputs during OPSD.
    """
    cursor = cache["position"]
    if start * tokens_per_frame != cursor.end:
        raise ValueError(f"ABot cache ends at frame {cursor.end // tokens_per_frame}, requested {start}")
    if commit and torch.is_grad_enabled():
        raise RuntimeError("ABot history commits must run under no_grad")
    ref = reference_tokens
    query_ref = ref if start == 0 else 0
    capacity = (cache["k"].shape[1] - ref) // tokens_per_frame
    if frames > capacity:
        raise ValueError("ABot block exceeds its local attention window")
    sink = min(cache.get('sink_tokens', 0) // tokens_per_frame, cursor.length // tokens_per_frame)
    rolling = min(cursor.length // tokens_per_frame - sink, capacity - frames - sink)
    if rolling < 0:
        raise ValueError("ABot block exceeds the rolling region after reserving its sink")
    retained = sink + rolling
    sink_end = ref + sink * tokens_per_frame
    past_end = ref + cursor.length
    past_start = past_end - rolling * tokens_per_frame
    raw = torch.cat((cache["k_raw"][:, ref:sink_end], cache["k_raw"][:, past_start:past_end], k[:, query_ref:]), 1)
    values = torch.cat((cache["v"][:, ref:sink_end], cache["v"][:, past_start:past_end], v[:, query_ref:]), 1)
    base = cursor.rope_base // tokens_per_frame
    position = start - base
    # Official REL_ROPE_CACHE=1, REL_ROPE_REBASE_MAX_POS=256.
    if position + frames > ROPE_REBASE_FRAMES or position < 0:
        new_base = start - rolling
        position = start - new_base
        sink_keys = cache['k'][:, ref:sink_end].clone()
        if sink:
            from wan.kv_cache.ops import _rope_time_delta_mul_
            sink_keys = rope_video(raw[:, :sink * tokens_per_frame], sink, 0)
            delta = -new_base
            while delta:
                shift = max(1 - len(freqs), min(len(freqs) - 1, delta))
                _rope_time_delta_mul_(sink_keys, freqs, shift)
                delta -= shift
        rolling_keys = rope_video(raw[:, sink * tokens_per_frame:], rolling + frames, position - rolling)
        keys = torch.cat((sink_keys, rolling_keys), 1)
        base = new_base
    else:
        keys = torch.cat((cache["k"][:, ref:sink_end], cache["k"][:, past_start:past_end],
                          rope_video(k[:, query_ref:], frames, position)), 1)
    query = rope_video(q[:, query_ref:], frames, position)
    ref_raw = k[:, :ref] if query_ref else cache["k_raw"][:, :ref]
    ref_values = v[:, :ref] if query_ref else cache["v"][:, :ref]
    attn_keys = torch.cat((rope_reference(ref_raw), keys), 1)
    attn_values = torch.cat((ref_values, values), 1)
    if query_ref:
        query = torch.cat((rope_reference(q[:, :ref]), query), 1)
    key_lengths = None
    output = attend(query, attn_keys, attn_values, k_lens=key_lengths)
    if commit:
        end = ref + (retained + frames) * tokens_per_frame
        if query_ref:
            cache["k_raw"][:, :ref].copy_(ref_raw)
            cache["v"][:, :ref].copy_(ref_values)
        cache["k_raw"][:, ref:end].copy_(raw)
        cache["k"][:, ref:end].copy_(keys)
        cache["v"][:, ref:end].copy_(values)
        cursor.length = (retained + frames) * tokens_per_frame
        cursor.end = (start + frames) * tokens_per_frame
        cursor.rope_base = base * tokens_per_frame
    return output


ACTION_KEYS = ("W", "A", "S", "D", "I", "J", "K", "L")
INTERACTIVE_ACTIONS = {
    "WAIT": None,
    "W": "W", "A": "A", "S": "S", "D": "D",
    "UP": "I", "LEFT": "J", "DOWN": "K", "RIGHT": "L",
    "I": "I", "J": "J", "K": "K", "L": "L",
}


def abot_action_state(action, *, device=None, dtype=torch.bfloat16):
    """Map a PlayWorld action to one native WASDIJKL state vector."""
    name = str(action).strip().upper()
    if name not in INTERACTIVE_ACTIONS:
        raise ValueError(
            f"Unsupported ABot action {action!r}; expected "
            + ", ".join(INTERACTIVE_ACTIONS)
        )
    state = torch.zeros(len(ACTION_KEYS), device=device, dtype=dtype)
    native = INTERACTIVE_ACTIONS[name]
    if native is not None:
        state[ACTION_KEYS.index(native)] = 1
    return state


REFERENCE_SLOTS = ("head", "left", "right", "front", "back")


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        return out


class SimpleAdapter(nn.Module):
    """ABot's released PixelUnshuffle/Conv2d/residual adapter, unchanged."""

    def __init__(self, in_dim, out_dim, kernel_size, stride, downscale_factor=16):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv = nn.Conv2d(
            in_dim * downscale_factor * downscale_factor, out_dim,
            kernel_size=kernel_size, stride=stride, padding=0,
        )
        self.residual_blocks = nn.Sequential(ResidualBlock(out_dim))

    def forward(self, x):
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        x = self.conv(self.pixel_unshuffle(x))
        out = self.residual_blocks(x)
        return out.view(bs, f, *out.shape[1:]).permute(0, 2, 1, 3, 4)


def action_features(adapter, actions, height, width, checkpoint_func=None):
    """Encode each distinct key state once, without a full video-sized RGB map.

    Actions are spatially constant in the official set_act interface. Reusing
    an identical frame's adapter output preserves its spatial convolutions and
    padding; summing convolution weights would change BF16 rounding.
    """
    states, inverse = torch.unique(actions.flatten(0, 1), dim=0, return_inverse=True)

    def encode(state):
        pixels = state.repeat_interleave(4).view(1, 32, 1, 1, 1)
        return adapter(pixels.expand(-1, -1, -1, height, width))[0, :, 0]

    features = torch.stack([
        checkpoint_func(encode, state)
        if checkpoint_func is not None and torch.is_grad_enabled() else encode(state)
        for state in states
    ])
    return features[inverse].unflatten(0, actions.shape[:2]).transpose(1, 2)


class ActionAdapter(SimpleAdapter):
    """One FSDP call per sequence, regardless of its number of distinct actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-state loops must not issue a rank-dependent number of collectives.
        for module in self.modules():
            module._fsdp_wrap = module is self

    def forward(self, actions, height, width, checkpoint_func=None):
        return action_features(super().forward, actions, height, width, checkpoint_func)


class ABotCrossAttention(WanSelfAttention):
    def forward(self, x, context, cache=None):
        b, n, d = x.shape[0], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        # Training always computes text K/V with gradients, including OPSD.
        if cache is None or torch.is_grad_enabled() or not cache.get("is_init"):
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            if cache is not None and not torch.is_grad_enabled():
                cache.update(k=k, v=v, is_init=True)
        else:
            k, v = cache["k"], cache["v"]
        return self.o(attention(q, k, v).flatten(2))


class ABotSelfAttention(WanSelfAttention):
    def forward(self, x, grid, freqs, ref_grid, *, mask=None, cache=None,
                current_start=0, commit=False):
        q, k, v = self.project_qkv(x)
        frames, h, w = grid[0].tolist()
        tpf = h * w
        slots, rt, rh, rw = ref_grid
        ref = slots * rt * rh * rw
        ref_freqs = _build_ref_freqs(freqs, slots, rt * rh * rw, (rt, rh, rw), x.device)

        def rotate_ref(value):
            return rope_apply_with_refimg(value, ref_freqs, self.num_heads).type_as(v)

        def rotate_video(value, count, start):
            video_grid = grid.clone()
            video_grid[:, 0] = count
            return causal_rope_apply(value, video_grid, freqs, start).type_as(v)

        if cache is None:
            branch = frames * tpf
            def rotate_training(value):
                return torch.cat((rotate_ref(value[:, :ref]),
                                  rotate_video(value[:, ref:ref + branch], frames, 0),
                                  rotate_video(value[:, ref + branch:], frames, 0)), 1)
            output = CausalWanSelfAttention._padded_flex_attention(
                rotate_training(q), rotate_training(k), v, mask,
            )
        else:
            if current_start % tpf:
                raise ValueError("ABot current_start must be a latent-frame boundary")
            output = cache_attention(
                q, k, v, cache, start=current_start // tpf, frames=frames,
                tokens_per_frame=tpf, reference_tokens=ref, rope_video=rotate_video,
                rope_reference=rotate_ref, attend=attention, commit=commit, freqs=freqs,
            )
        return self.o(output.flatten(2))


class ABotBlock(WanAttentionBlock):
    self_attention_type = ABotSelfAttention

    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=True, eps=1e-6,
                 cross_attention=ABotCrossAttention):
        super().__init__(dim, ffn_dim, num_heads, window_size, qk_norm,
                         cross_attn_norm, eps, cross_attention)

    def forward(self, x, e, context, *, cross_cache=None, **attention_kwargs):
        token_modulation = e.shape[1] == x.shape[1]
        frames = e.shape[1]
        tpf = x.shape[1] // frames
        shift, scale, gate, ffn_shift, ffn_scale, ffn_gate = (
            self.modulation.unsqueeze(0) + e
        ).chunk(6, dim=2)

        def modulate(value, scale, shift):
            if token_modulation:
                return value * (1 + scale.squeeze(2)) + shift.squeeze(2)
            return (value.unflatten(1, (frames, tpf)) * (1 + scale) + shift).flatten(1, 2)

        def gated(value, gate):
            if token_modulation:
                return value * gate.squeeze(2)
            return (value.unflatten(1, (frames, tpf)) * gate).flatten(1, 2)

        y = self.self_attn(modulate(self.norm1(x), scale, shift), **attention_kwargs)
        x = x + gated(y, gate)
        x = x + self.cross_attn(self.norm3(x), context, cross_cache)
        return x + gated(self.ffn(modulate(self.norm2(x), ffn_scale, ffn_shift)), ffn_gate)


class ABotHead(Head):
    def forward(self, x, e):
        frames = e.shape[1]
        shift, scale = (self.modulation.unsqueeze(1) + e.unsqueeze(2)).chunk(2, dim=2)
        x = self.norm(x).unflatten(1, (frames, x.shape[1] // frames))
        return self.head(x * (1 + scale) + shift).flatten(1, 2)


def teacher_forcing_mask(frames, tpf, refs, block, device):
    """Official reference/clean/noisy mask, with the independent first frame.

    Reference queries see reference slots only. A noisy block sees references,
    earlier clean blocks and itself, never its own or future clean targets.
    The released teacher-forcing path uses full clean history.
    """
    branch = frames * tpf
    clean_offset, noisy_offset = refs, refs + branch
    total = refs + 2 * branch
    padded = math.ceil(total / 128) * 128
    ends = torch.zeros(padded, device=device, dtype=torch.long)
    history_ends = torch.zeros_like(ends)
    noisy_starts = torch.zeros_like(ends)
    noisy_ends = torch.zeros_like(ends)
    ranges = [(0, 1)] + [(start, min(start + block, frames)) for start in range(1, frames, block)]
    for start, end in ranges:
        start, end = start * tpf, end * tpf
        ends[clean_offset + start:clean_offset + end] = clean_offset + end
        history_ends[noisy_offset + start:noisy_offset + end] = clean_offset + start
        noisy_starts[noisy_offset + start:noisy_offset + end] = noisy_offset + start
        noisy_ends[noisy_offset + start:noisy_offset + end] = noisy_offset + end

    def visible(b, h, q, k):
        ref_q, ref_k = q < refs, k < refs
        clean_q = (q >= clean_offset) & (q < noisy_offset)
        noisy_q = (q >= noisy_offset) & (q < total)
        clean = clean_q & (k >= clean_offset) & (k < ends[q])
        noisy = noisy_q & (k >= noisy_starts[q]) & (k < noisy_ends[q])
        history = noisy_q & (k >= clean_offset) & (k < history_ends[q])
        return (ref_q & ref_k) | (~ref_q & (ref_k | clean | noisy | history)) | (q == k)

    return create_block_mask(visible, B=None, H=None, Q_LEN=padded,
                             KV_LEN=padded, device=device, _compile=True)


class ABotDiT(WanModel):
    """Keep native Wan2.2 parameter names; specialize ABot conditions and AR math."""

    supported_model_types = ("ci2v",)
    rope_rebase_frames = ROPE_REBASE_FRAMES
    block_type = ABotBlock
    head_type = ABotHead
    _no_split_modules = ["ABotBlock"]

    @register_to_config
    def __init__(self, model_type="ci2v", patch_size=(1, 2, 2), text_len=512,
                 in_dim=48, dim=3072, ffn_dim=14336, freq_dim=256, text_dim=4096,
                 out_dim=48, num_heads=24, num_layers=30, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=True, eps=1e-6,
                 act_control_in_dim=32, downscale_factor_control_adapter=16,
                 num_frame_per_block=3, local_attn_size=21, sink_size=0,
                 use_relative_rope=True):
        if sink_size != 0 or not use_relative_rope or local_attn_size != 21:
            raise ValueError("ABot-World-0 requires relative RoPE, window=21 and video sink=0")
        self.act_control_in_dim = act_control_in_dim
        self.downscale_factor_control_adapter = downscale_factor_control_adapter
        self.num_frame_per_block = num_frame_per_block
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self._mask_cache = {}
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim,
                         freq_dim, text_dim, out_dim, num_heads, num_layers,
                         window_size, qk_norm, cross_attn_norm, eps)

    def cross_attention_type(self, model_type):
        return ABotCrossAttention

    def init_conditioning(self):
        self.act_control_adapter = ActionAdapter(
            self.act_control_in_dim, self.dim, self.patch_size[1:], self.patch_size[1:],
            self.downscale_factor_control_adapter,
        )

    def _time(self, t):
        e, e0 = self.embed_timesteps(t)
        return e, e0.unflatten(2, (6, self.dim))

    def _reference_tokens(self, references, mask):
        b, slots, c, f, h, w = references.shape
        ref = self.patch_embedding(references.detach().reshape(b * slots, c, f, h, w))
        grid = (slots, *ref.shape[2:])
        ref = ref.flatten(2).transpose(1, 2).unflatten(0, (b, slots))
        return (ref * mask.detach()[:, :, None, None]).flatten(1, 2), grid

    def forward(self, x, t, context, *, actions, reference_latent, reference_mask, clean_x=None, kv_cache=None,
                crossattn_cache=None, current_start=0, commit_cache=False):
        """Input and output are [B,C,F,H,W]; timesteps are [B,F]."""
        b, _, frames, height, width = x.shape
        if actions.shape != (b, frames, 8):
            raise ValueError("ABot actions must align with every input latent frame")
        self.freqs = self.freqs.to(x.device)
        action = self.act_control_adapter(
            actions,
            height * self.downscale_factor_control_adapter,
            width * self.downscale_factor_control_adapter,
            checkpoint_func=self._gradient_checkpointing_func if self.gradient_checkpointing else None,
        )
        tokens = self.patch_embedding(x) + action
        grid = torch.tensor(tokens.shape[2:], device=x.device)[None].expand(b, -1)
        tpf = tokens.shape[3] * tokens.shape[4]
        tokens = tokens.flatten(2).transpose(1, 2)
        e, e0 = self._time(t)
        context = self.project_context(context, None)
        refs = reference_latent
        ref_grid = (refs.shape[1], refs.shape[3] // self.patch_size[0],
                    refs.shape[4] // self.patch_size[1], refs.shape[5] // self.patch_size[2])
        include_ref = kv_cache is None or current_start == 0
        mask = None
        if kv_cache is None:
            if clean_x is None or clean_x.shape != x.shape:
                raise ValueError("ABot SFT requires aligned clean/noisy teacher-forcing branches")
            clean = (self.patch_embedding(clean_x) + action).flatten(2).transpose(1, 2)
            _, clean_e0 = self._time(torch.zeros_like(t))
            tokens = torch.cat((clean, tokens), 1)
            e0 = torch.cat((clean_e0, e0), 1)
        if include_ref:
            ref_tokens, ref_grid = self._reference_tokens(refs, reference_mask)
            _, ref_e0 = self._time(t.new_zeros((b, 1)))
            e0 = torch.cat((ref_e0.expand(-1, ref_tokens.shape[1], -1, -1),
                            e0.repeat_interleave(tpf, 1)), 1)
            tokens = torch.cat((ref_tokens, tokens), 1)
        if kv_cache is None:
            mask_key = (frames, tpf, ref_grid, str(x.device))
            if mask_key not in self._mask_cache:
                self._mask_cache[mask_key] = teacher_forcing_mask(
                    frames, tpf, ref_tokens.shape[1], self.num_frame_per_block, x.device,
                )
            mask = self._mask_cache[mask_key]
        for i, block in enumerate(self.blocks):
            kwargs = dict(grid=grid, freqs=self.freqs, ref_grid=ref_grid, mask=mask,
                          cache=None if kv_cache is None else kv_cache[i],
                          cross_cache=None if crossattn_cache is None else crossattn_cache[i],
                          current_start=current_start, commit=commit_cache)
            if self.gradient_checkpointing and torch.is_grad_enabled():
                tokens = self._gradient_checkpointing_func(partial(block, **kwargs), tokens, e0, context)
            else:
                tokens = block(tokens, e0, context, **kwargs)
        tokens = tokens[:, -frames * tpf:]
        return torch.stack(self.unpatchify(self.head(tokens, e), grid))


class ABotModel(WanModelWrapper):
    """Expose ABot flow prediction and caches to the common SFT/OPSD engines."""

    def __init__(self, config):
        super().__init__(ABotDiT(**config))

    def forward(self, latent, timestep, context, *, actions, reference_latent, reference_mask,
                clean_latent=None, kv_cache=None, crossattn_cache=None,
                current_start=0, position_start=None,
                extra_sink_size=0, commit_cache=False, **text_caches):
        if extra_sink_size:
            raise ValueError("ABot's temporal image prefix is not a persistent sink")
        if position_start not in (None, current_start):
            raise ValueError("ABot manages relative RoPE positions inside its KV cache")
        from utils import model_autocast
        with model_autocast(self.model):
            result = self.guided_forward(self.model,
                latent.transpose(1, 2), timestep, context=context, actions=actions,
                reference_latent=reference_latent, reference_mask=reference_mask,
                clean_x=None if clean_latent is None else clean_latent.transpose(1, 2),
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=current_start, commit_cache=commit_cache,
                **text_caches,
            )
        return result.transpose(1, 2).float()

    def new_cache(self, batch, frames, height, width, device, dtype, extra_sink_size=0,
                  *, capacity_frames=None, sink_frames=None):
        # Reference slots remain separate from configurable temporal sinks.
        from wan.kv_cache.causal import CausalCache, cross_attention_cache

        return self.guided_cache(lambda: CausalCache({"kv_cache": allocate(self.model, batch, frames, height, width, device, dtype,
                                                capacity_frames=capacity_frames,
                                                sink_frames=sink_frames)},
                           cross_attention_cache(self.model), reference_tokens=1280))


class ABotWorld(WanAR):
    camera_modules = ("act_control_adapter",)
    base = "wan2.2"
    independent_first_frame = True
    image_condition_sink = False
    required_data_fields = ("actions",)
    interactive_control_features = len(ACTION_KEYS)
    output_includes_condition = True
    inference_frames_include_condition = True
    num_frame_per_block = 3
    local_attn_size = 21
    sink_size = 0
    camera_control = False  # Discrete actions are handled here, not as calibrated poses.
    player_rotation_degrees = 24.0

    @staticmethod
    def interactive_prompt(prompt):
        value = str(prompt).strip()
        if not value:
            raise ValueError("Interactive prompt cannot be empty")
        return value if value.startswith("|") else f"| unknown | {value}"

    default_latent_frames = 31
    sampling_steps = (1000, 750, 500, 250)
    timestep_shift = 5.0
    latent_shape = (48, 44, 80)
    pixel_shape = (704, 1280)

    def validate_sampling_steps(self, sampling_steps):
        return parse_sampling_steps(sampling_steps)

    def opsd_schedule(self, device, sampling_steps, *, timestep_shift=None, sampling_mode=None):
        return make_flow_match_scheduler(
            device, self.validate_sampling_steps(sampling_steps),
            timestep_shift=self.timestep_shift if timestep_shift is None else timestep_shift,
            sampling_mode=self.sampling_mode if sampling_mode is None else sampling_mode,
        )

    @staticmethod
    def checkpoint_file(checkpoint):
        checkpoint = Path(checkpoint)
        if checkpoint.is_dir() or not checkpoint.suffix:
            return checkpoint / "diffusion_pytorch_model.safetensors"
        if checkpoint.suffix == ".safetensors" or checkpoint.name == "model.pt":
            return checkpoint
        raise ValueError("ABot checkpoint must be its official model directory/safetensors or project model.pt")

    def validate(self, model_dir, checkpoint, *, codec=False, config=None):
        model_dir = Path(model_dir)
        required = [model_dir / "config.json", model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
                    self.checkpoint_file(checkpoint)]
        if codec:
            required.append(model_dir / "Wan2.2_VAE.pth")
        missing = [str(path) for path in required if not path.is_file()]
        if not (model_dir / "google/umt5-xxl").is_dir():
            missing.append(str(model_dir / "google/umt5-xxl"))
        if missing:
            raise FileNotFoundError("Missing ABot assets:\n" + "\n".join(missing))
        self._official_config(model_dir)

    @staticmethod
    def _official_config(model_dir):
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        expected = dict(dim=3072, ffn_dim=14336, freq_dim=256, in_dim=48, out_dim=48,
                        num_heads=24, num_layers=30, text_len=512, eps=1e-6,
                        downscale_factor_control_adapter=16)
        mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
        if mismatches:
            raise ValueError(f"Not the released ABot-World-0-5B-LF configuration (actual, expected): {mismatches}")
        # The released config says ti2v; the official wrapper explicitly selects ci2v.
        return {**expected, "model_type": "ci2v"}

    def build(self, model_dir, config):
        self.validate_model_config(config)
        return ABotModel(self._official_config(model_dir))

    def transformer_layer(self):
        return ABotBlock


    def prepare_inputs(self, sample, target_frames, device, *, dtype=torch.bfloat16):
        """Prepare temporal actions and static reference slots for every execution path."""
        prefix = sample["condition_latent"].shape[-4]
        actions, condition_actions = sample["actions"], sample["condition_actions"]
        batched = actions.ndim == 3
        if not batched:
            actions, condition_actions = actions[None], condition_actions[None]
        if actions.shape[-1] != 8 or condition_actions.shape != (actions.shape[0], sample["condition_latent"].shape[-4], 8):
            raise ValueError("ABot needs eight action keys aligned with target and condition latents")
        actions = torch.cat((condition_actions, actions), 1)
        if not ((actions == 0) | (actions == 1)).all():
            raise ValueError("ABot action inputs must be binary W,A,S,D,I,J,K,L states")
        references = sample.get("reference_latent")
        mask = sample.get("reference_mask")
        if references is None:
            references = actions.new_zeros((actions.shape[0], 5, 48, 1, 32, 32))
            mask = actions.new_zeros((actions.shape[0], 5))
        else:
            if not batched:
                references, mask = references[None], mask[None]
            if references.shape != (actions.shape[0], 5, 48, 1, 32, 32) or mask.shape != (actions.shape[0], 5):
                raise ValueError("ABot requires five reference slots [5,48,1,32,32] and five mask values")
        if actions.shape[1] < prefix + target_frames:
            raise ValueError("ABot actions do not cover the requested latent timeline")
        return TemporalInputs(
            {"actions": self._move(actions, device)},
            {"reference_latent": self._move(references, device), "reference_mask": self._move(mask, device)},
        ).slice(0, prefix + target_frames)

    def fixed_view_inputs(self, inputs, prefix_frames, target_frames):
        if prefix_frames == 0:
            raise ValueError("ABot fixed-view OPSD requires an image or video condition")
        actions = inputs.temporal["actions"].clone()
        # The teacher releases every action key throughout its own history rollout.
        actions[:, prefix_frames:prefix_frames + target_frames] = 0
        return TemporalInputs({"actions": actions}, inputs.static)

    def add_inference_arguments(self, parser):

        parser.set_defaults(fps=12)  # Official web_client/config.py VIDEO_FPS.
        self.add_cache_arguments(parser)
        self.add_length_arguments(parser)
        parser.add_argument("--sampling-steps", type=parse_sampling_steps, default=None)
        parser.add_argument("--sampling-mode", choices=("ode", "sde"), default=None)
        parser.add_argument("--timestep-shift", type=float, default=None)

    def inference_options(self, args):

        steps = self.sampling_steps if args.sampling_steps is None else args.sampling_steps
        shift = getattr(args, "timestep_shift", None)
        shift = self.timestep_shift if shift is None else shift
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("timestep_shift must be finite and positive")
        return {**self.inference_length_options(args), "sampling_steps": self.validate_sampling_steps(steps),
                "sampling_mode": args.sampling_mode or self.sampling_mode,
                "timestep_shift": shift,

                "sink_frames": getattr(args, 'sink_frames', None)}

    interactive_action_state = staticmethod(abot_action_state)

    @torch.no_grad()
    def _inference(self, model, samples, text_encoder, codec, seeds, device, **options):
        from sampling import denoise_few_step_block, sample_causal_sequence

        configured_frames = options["num_latent_frames"]
        contexts, negative_contexts = self.inference_contexts(samples, text_encoder, device, **options)
        text_scale = self.inference_text_options(options)['guidance_scale']
        outputs, images = [], []
        # Each sample keeps its own official RNG/action/reference/cache timeline.
        for index, (row, seed, context) in enumerate(zip(samples, seeds, contexts, strict=True)):
            prefix = row["condition_latent"]
            count = len(prefix)
            frames = (
                count + int(row["target_frames"])
                if configured_frames is None else configured_frames
            )
            if frames <= 1 or (frames - 1) % self.num_frame_per_block:
                raise ValueError(
                    "ABot data length must contain 1 + 3n total latent frames"
                )
            if count >= frames:
                raise ValueError("ABot output must extend beyond the condition")
            if prefix.shape[1:] != self.latent_shape:
                raise ValueError(f"ABot expects latent frame shape {self.latent_shape}")
            inputs = self.prepare_inputs(row, frames - count, device)
            prefix = self._move(prefix[None], device)
            noise, generators = self.initial_noise([seed], (frames, *self.latent_shape), device, 1.0)
            generator = generators[0]
            negative = None if negative_contexts is None else [negative_contexts[index]]
            with self.unwrap(model).text_guidance(negative, text_scale):
                cache = model.new_cache(1, frames, *self.latent_shape[1:], device, torch.bfloat16,
                                         sink_frames=options.get('sink_frames'))
                scheduler = self.inference_schedule(device, options)
                tpf = self.latent_shape[1] // 2 * (self.latent_shape[2] // 2)

                def prepare_block(start, length):
                    block_inputs = inputs.slice(start, length).forward_kwargs()

                    def predict_flow(latent, step):
                        return model(latent, step.float().expand(1, length), [context], **block_inputs,
                                     **cache.forward_kwargs(), current_start=start * tpf)

                    def commit(clean):
                        model(clean, clean.new_zeros((1, length)), [context], **block_inputs,
                              **cache.forward_kwargs(), current_start=start * tpf, commit_cache=True)

                    return predict_flow, commit

                lengths = self.temporal_layout.target_blocks(frames - count, prefix_frames=count)
                def draw_noise(latent):
                    return torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)

                def denoise(latent, predict):
                    return denoise_few_step_block(latent, scheduler, predict, draw_noise)

                outputs.append(sample_causal_sequence(
                    noise[:, count:], lengths,
                    prepare_block, denoise, prefix=prefix, prefix_lengths=self.condition_block_lengths(count),
                    include_prefix=True, progress="ABot blocks",
                ))
            images.append(count > 0)
            cache = inputs = None
        latent = torch.cat(outputs)
        video = (self.decode(codec, latent) * 0.5 + 0.5).clamp(0, 1)
        return video, latent, video[:, 0].clone() if all(images) else None

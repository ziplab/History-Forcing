"""ReWorld's Pluecker injection and pose-modulated causal Wan2.2 architecture."""
import math

import torch
from torch import nn

from wan.attention import attention
from wan.causal import (CausalWanModel, CausalWanAttentionBlock,
                        CausalWanSelfAttention, causal_rope_apply)
from wan.geometry import _invert_SE3
from wan.kv_cache.causal import read_attention, update_cache
from wan.wan21.model import MLPProj

from .base import InferenceLora
from .wan_ar import Action2VDMDGenerator, WanAR


def pose_apply(x, poses, inverse=False):
    """Released block-diagonal SE3 transform on each four head channels."""
    b, s, heads, dim = x.shape
    frames = poses.shape[1]
    transforms = _invert_SE3(poses) if inverse else poses
    values = x.permute(0, 2, 1, 3).reshape(b, heads, frames, s // frames, dim // 4, 4)
    result = torch.matmul(transforms[:, None, :, None, None].to(x.dtype), values.unsqueeze(-1))
    return result.squeeze(-1).reshape(b, heads, s, dim).permute(0, 2, 1, 3)


def phase_apply(x, delta):
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotation = torch.polar(torch.ones_like(delta.float()), delta.float())
    return torch.view_as_real(complex_x * rotation).flatten(-2).to(x.dtype)


def plucker_tokens(poses, height, width):
    """Native plucker_720p: normalized focal 1, half-pixel centers, [d,m]."""
    with torch.autocast(poses.device.type, enabled=False):
        poses = poses.float()
        y, x = torch.meshgrid(torch.arange(height, device=poses.device, dtype=torch.float32),
                              torch.arange(width, device=poses.device, dtype=torch.float32), indexing='ij')
        rays = torch.stack(((x + .5 - width / 2) / width, (y + .5 - height / 2) / height, torch.ones_like(x)), -1)
        rays = rays / (rays.norm(dim=-1, keepdim=True) + 1e-8)
        directions = torch.einsum('bfij,hwj->bfhwi', poses[..., :3, :3], rays)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        moments = torch.cross(poses[..., None, None, :3, 3].expand_as(directions), directions, dim=-1)
        return torch.cat((directions, moments), -1)


class ReWorldAttention(CausalWanSelfAttention):
    def __init__(self, native, local_attn_size=-1, sink_size=0):
        super().__init__(native, local_attn_size, sink_size)
        self.pose_mlp = nn.Sequential(nn.Linear(12, 256), nn.SiLU(), nn.Linear(256, self.head_dim // 2))
        self.value_proj = nn.Linear(self.dim, self.dim)
        self.prope_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, seq_lens, grid_sizes, freqs, *, poses, block_mask=None,
                kv_cache=None, current_start=0, position_start=None, commit_cache=True,
                extra_sink_size=0, **kwargs):
        s = x.shape[1]
        q, k, v = self.project_qkv(x)
        tpf = math.prod(grid_sizes[0, 1:].tolist())
        descriptor = torch.cat((poses[..., :3, :3].flatten(-2), poses[..., :3, 3]), -1)
        delta = self.pose_mlp(descriptor).repeat_interleave(tpf, dim=1)[:, :, None]
        v = (v.flatten(2) + self.value_proj(pose_apply(v, poses, inverse=True).flatten(2))).view_as(q)
        if kv_cache is None:
            target = int(seq_lens[0])
            prefix = s - 2 * target
            parts_q, parts_k = [], []
            for begin, length, frame in ((0, prefix, 0), (prefix, target, prefix // tpf),
                                        (prefix + target, target, prefix // tpf)):
                if not length:
                    continue
                grid = grid_sizes.clone()
                grid[:, 0] = length // tpf
                offset = delta[:, begin:begin+length]
                parts_q.append(phase_apply(causal_rope_apply(q[:, begin:begin+length], grid, freqs, frame).type_as(v), offset))
                parts_k.append(phase_apply(causal_rope_apply(k[:, begin:begin+length], grid, freqs, frame).type_as(v), offset))
            rq, rk = torch.cat(parts_q, 1), torch.cat(parts_k, 1)
            result = torch.empty_like(v)
            for heads, mask in block_mask:
                result[:, :, heads] = self._padded_flex_attention(rq[:, :, heads], rk[:, :, heads], v[:, :, heads], mask)
        else:
            frame = (current_start if position_start is None else position_start) // tpf
            rq = phase_apply(causal_rope_apply(q, grid_sizes, freqs, frame).type_as(v), delta)
            rk = phase_apply(causal_rope_apply(k, grid_sizes, freqs, frame).type_as(v), delta)
            sink = (self.sink_size + extra_sink_size) * tpf
            keys, values, lengths = read_attention(kv_cache, rk, v, current_start, sink, tpf, freqs=freqs)
            result = attention(rq, keys, values, k_lens=lengths)
            if commit_cache:
                update_cache(kv_cache, rk.detach(), v.detach(), current_start,
                             sink)
        return self.o(result.flatten(2)) + self.prope_proj(pose_apply(result, poses).flatten(2))


class ReWorldBlock(CausalWanAttentionBlock):
    attention_type = ReWorldAttention

    def __init__(self, native, local_attn_size=-1, sink_size=0):
        super().__init__(native, local_attn_size, sink_size)
        self.action_embedding = MLPProj(6, native.self_attn.dim)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, *,
                poses, action_maps, action_start=0, block_mask=None, kv_cache=None,
                crossattn_cache=None, commit_cache=True, secondary_context=None,
                context_mask=None, **kwargs):
        frames = e.shape[1]
        m = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        inputs = (self.norm1(x).unflatten(1, (frames, -1)) * (1 + m[1]) + m[0]).flatten(1, 2)
        # Native Step 3.3 writes clean pose-conditioned KV without action IDs.
        if kv_cache is None or not commit_cache:
            action = self.action_embedding(action_maps).flatten(1, 3)
            inputs = torch.cat((inputs[:, :action_start], inputs[:, action_start:] + action), 1)
        residual = self.self_attn(inputs, seq_lens, grid_sizes, freqs, poses=poses,
                                 block_mask=block_mask, kv_cache=kv_cache, commit_cache=commit_cache, **kwargs)
        x = x + (residual.unflatten(1, (frames, -1)) * m[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache,
                               secondary_context=secondary_context, context_mask=context_mask)
        residual = self.ffn((self.norm2(x).unflatten(1, (frames, -1)) * (1 + m[4]) + m[3]).flatten(1, 2))
        return x + (residual.unflatten(1, (frames, -1)) * m[5]).flatten(1, 2)


class ReWorldModel(CausalWanModel):
    base = 'wan2.2'
    block_type = ReWorldBlock
    _no_split_modules = ['ReWorldBlock']

    def _init_conditioning(self, control_type):
        if control_type is not None:
            raise ValueError('ReWorld camera conditioning is part of its architecture')
        d = self.dim // self.num_heads // 2
        temporal, height, width = self.freqs.split((d - 2 * (d // 3), d // 3, d // 3), -1)
        positions = torch.arange(len(self.freqs), device=self.freqs.device, dtype=torch.float64)[:, None]
        height = torch.polar(torch.ones_like(height.real), positions * torch.angle(height[1:2]) * (12 / 22))
        width = torch.polar(torch.ones_like(width.real), positions * torch.angle(width[1:2]) * (20 / 40))
        self.freqs = torch.cat((temporal, height, width), -1)

    def block_kwargs(self, tokens, grid_sizes, *, viewmats, intrinsics,
                     prefix_viewmats=None, prefix_intrinsics=None,
                     prefix_frames=0, teacher_forcing=False):
        poses = _invert_SE3(viewmats)
        h, w = grid_sizes[0, 1:].tolist()
        maps = plucker_tokens(poses, h, w).to(tokens.dtype)
        action_start = 0
        if teacher_forcing:
            prefix = poses[:, :0] if not prefix_frames else _invert_SE3(prefix_viewmats)
            poses = torch.cat((prefix, poses, poses), 1)
            action_start = (prefix_frames + grid_sizes[0, 0].item()) * h * w
        return dict(poses=poses, action_maps=maps, action_start=action_start)

    def _get_teacher_mask(self, device, frames, tokens_per_frame, prefix_frames):
        key = (device, frames, tokens_per_frame, prefix_frames)
        if key not in self._teacher_masks:
            global_heads = [i for i in range(self.num_heads) if i % 5 == 0 or i == 23]
            groups = ((global_heads, -1), ([i for i in range(self.num_heads) if i not in global_heads], 12))
            self._teacher_masks[key] = [(heads, self._teacher_forcing_mask(
                device, frames, tokens_per_frame,
                self.temporal_layout.target_blocks(frames, prefix_frames=prefix_frames),
                self.temporal_layout.prefix_blocks(prefix_frames), prefix_frames, window,
                0, self.temporal_layout.extra_sink_frames(prefix_frames))) for heads, window in groups if heads]
        return self._teacher_masks[key]


class ReWorld(Action2VDMDGenerator, WanAR):
    """Released architecture with shared UniPC rollout, recipes and Player."""
    camera_modules = (
        'self_attn.pose_mlp', 'self_attn.value_proj',
        'self_attn.prope_proj', 'action_embedding',
    )
    base = 'wan2.2'
    camera_control = True
    required_data_fields = ('viewmats', 'intrinsics')
    num_frame_per_block = 4
    local_attn_size = -1
    sink_size = 0
    default_latent_frames = None  # Use each packed camera/task length.
    output_includes_condition = True
    latent_shape = (48, 44, 80)
    pixel_shape = (704, 1280)
    sampling_steps = 4
    sampling_mode = 'ode'
    player_translation_step = 4 * .35
    player_rotation_degrees = math.degrees(4 * .0625)
    player_pitch_sign = -1
    player_yaw_space = 'world'
    playworld_max_video_seconds = 20.0

    @staticmethod
    def transformer_class():
        return ReWorldModel

    def add_inference_arguments(self, parser):
        super().add_inference_arguments(parser)
        parser.set_defaults(fps=24, seed=0)

    def inference_lora(self, model, checkpoint):
        checkpoint = self.checkpoint_file(checkpoint)
        if checkpoint.name == 'model.pt':
            return None
        path = checkpoint.with_name('reworld_5b_dmd_lora.pt')
        if not path.is_file():
            raise FileNotFoundError(f'Missing released DMD LoRA: {path}')
        state = torch.load(path, map_location='meta', mmap=True, weights_only=True)['generator_lora']
        ranks = {value.shape[0] for name, value in state.items() if name.endswith('.lora_A.weight')}
        if len(ranks) != 1:
            raise ValueError('Released ReWorld LoRA ranks differ')
        return InferenceLora(path, ('q', 'k', 'v', 'o', 'ffn.0', 'ffn.2'), ranks.pop(), 128)

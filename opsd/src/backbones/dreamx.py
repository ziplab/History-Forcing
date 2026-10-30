"""DreamX AR and full-context Wan2.2 backends, using the shared model contracts.

Adapted from AMAP-ML/DreamX-World.
The release has separate visual and camera projections. AR camera attention is
chunk-local: the official pipeline allocates only visual KV, not prope_k/v.
"""
import math

import torch
from torch import nn

from wan.attention import attention
from wan.causal import CausalWanModel, CausalWanAttentionBlock, CausalWanSelfAttention, indexed_rope_apply
from wan.geometry import _invert_SE3, projective_qkv
from wan.kv_cache.causal import read_attention, update_cache
from wan.layers import WanAttentionBlock, WanRMSNorm
from wan.wan22.model import WanModel as Wan22Model

from .base import WanBackend
from .wan import WanFullContext, Wan22TI2V
from .wan_ar import Action2VDMDGenerator, WanAR

CAMERA_OPTIONS = ("add_control_adapter", "cam_method", "attn_compress", "cam_self_attn_layers")
NORMALIZED_K = ((969.6969696969696 / 1920, 0., .5),
                (0., 969.6969696969696 / 1080, .5), (0., 0., 1.))


class DreamXCameraAttention(nn.Module):
    def __init__(self, dim, attn_dim, heads, qk_norm, eps):
        super().__init__()
        if attn_dim % heads or (attn_dim // heads) % 4:
            raise ValueError("DreamX camera head dimension must be divisible by four")
        self.num_heads, self.head_dim = heads, attn_dim // heads
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)
        self.norm_q = WanRMSNorm(attn_dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(attn_dim, eps) if qk_norm else nn.Identity()
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, viewmats, Ks, *, chunk_tokens=None, seq_lens=None):
        b, s = x.shape[:2]
        heads, dim = self.num_heads, self.head_dim
        q = self.norm_q(self.q_proj(x)).view(b, s, heads, dim)
        k = self.norm_k(self.k_proj(x)).view(b, s, heads, dim)
        v = self.v_proj(x).view(b, s, heads, dim)
        # Parallel SFT keeps the released chunk-local camera visibility.
        if chunk_tokens is not None:
            if s % chunk_tokens:
                raise ValueError("Camera tokens must partition into complete AR blocks")
            groups = s // chunk_tokens
            q, k, v = (u.reshape(b * groups, chunk_tokens, heads, dim) for u in (q, k, v))
            viewmats = viewmats.reshape(b * groups, chunk_tokens, 4, 4)
            Ks = Ks.reshape(b * groups, chunk_tokens, 3, 3)
            seq_lens = None
        q, k, v, correct = projective_qkv(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            viewmats=viewmats, Ks=Ks,
        )
        output = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), k_lens=seq_lens)
        output = correct(output.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(output.flatten(2)).reshape(b, s, -1)


def _camera_tokens(tokens, grid_sizes, viewmats, Ks, *, teacher_forcing=False):
    tpf = math.prod(grid_sizes[0, 1:].tolist())
    views = viewmats.repeat_interleave(tpf, dim=1)
    intrinsics = Ks.repeat_interleave(tpf, dim=1)
    if teacher_forcing:
        views, intrinsics = (torch.cat((u, u), dim=1) for u in (views, intrinsics))
    if views.shape[1] != tokens.shape[1]:
        raise ValueError("DreamX camera and patch timelines differ")
    return dict(viewmats=views, Ks=intrinsics)


def _add_camera_branches(model, control, *, ar):
    control = control or {}
    if not control.get("add_control_adapter", True) or control.get("cam_method", "prope") != "prope":
        raise ValueError("DreamX released checkpoints require parallel PRoPE camera attention")
    compress = int(control.get("attn_compress", 4 if ar else 1))
    layers = control.get("cam_self_attn_layers")
    for index, block in enumerate(model.blocks):
        if layers is None or index in layers:
            heads = model.num_heads if ar else model.num_heads // compress
            block.cam_self_attn = DreamXCameraAttention(
                model.dim, model.dim // compress, heads, model.config.qk_norm, model.config.eps,
            )


class DreamXConfiguration:
    """Translate released camera config into the shared native conditioning hook."""
    conditioning_key = "conditioning_config"

    @classmethod
    def from_config(cls, config, **kwargs):
        config = dict(config)
        if any(key in config for key in CAMERA_OPTIONS):
            config[cls.conditioning_key] = {key: config.pop(key) for key in CAMERA_OPTIONS if key in config}
        return super().from_config(config, **kwargs)


class DreamXVisualAttention(CausalWanSelfAttention):
    def forward(self, x, seq_lens, grid_sizes, freqs, *, kv_cache=None,
                current_start=0, commit_cache=True, **kwargs):
        if kv_cache is None:
            return super().forward(x, seq_lens, grid_sizes, freqs, kv_cache=None, **kwargs)
        s = x.shape[1]
        q, k, v = self.project_qkv(x)
        tpf = math.prod(grid_sizes[0, 1:].tolist())
        sink = kv_cache.get('sink_tokens', self.sink_size * tpf)
        raw, values, lengths, positions = read_attention(
            kv_cache, k, v, current_start, sink, tpf,
            return_positions=True, relative_positions=True)
        end = raw.shape[1] if lengths is None else lengths.to(x.device)[:, None]
        query_ids = torch.arange(s, device=x.device) + end - s
        query_positions = positions.gather(1, query_ids.expand(x.shape[0], -1))
        roped_q = indexed_rope_apply(q, query_positions, grid_sizes, freqs).type_as(v)
        roped_k = indexed_rope_apply(raw, positions, grid_sizes, freqs).type_as(v)
        result = attention(roped_q, roped_k, values, k_lens=lengths)
        if commit_cache:
            update_cache(kv_cache, k.detach(), v.detach(), current_start, sink)
        return self.o(result.flatten(2))


class DreamXCausalBlock(CausalWanAttentionBlock):
    attention_type = DreamXVisualAttention

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                *, viewmats, Ks, block_mask=None, kv_cache=None, crossattn_cache=None,
                current_start=0, commit_cache=True, secondary_context=None,
                context_mask=None, **kwargs):
        frames = e.shape[1]
        m = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        normalized = self.norm1(x).unflatten(1, (frames, -1))
        inputs = (normalized * (1 + m[1]) + m[0]).flatten(1, 2)
        residual = self.self_attn(
            inputs, seq_lens, grid_sizes, freqs, block_mask=block_mask,
            kv_cache=kv_cache, current_start=current_start, commit_cache=commit_cache,
        )
        if hasattr(self, "cam_self_attn"):
            chunk = None if kv_cache is not None else 3 * math.prod(grid_sizes[0, 1:].tolist())
            residual = residual + self.cam_self_attn(inputs, viewmats, Ks, chunk_tokens=chunk)
        x = x + (residual.unflatten(1, (frames, -1)) * m[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens,
                               crossattn_cache=crossattn_cache, secondary_context=secondary_context,
                               context_mask=context_mask)
        residual = self.ffn((self.norm2(x).unflatten(1, (frames, -1)) * (1 + m[4]) + m[3]).flatten(1, 2))
        return x + (residual.unflatten(1, (frames, -1)) * m[5]).flatten(1, 2)


class DreamXCausalModel(DreamXConfiguration, CausalWanModel):
    base = "wan2.2"
    conditioning_key = "control_type"
    block_type = DreamXCausalBlock
    context_timestep = 0.1
    _no_split_modules = ["DreamXCausalBlock"]

    def _init_conditioning(self, control_type):
        _add_camera_branches(self, control_type, ar=True)
    def block_kwargs(self, tokens, grid_sizes, *, viewmats, intrinsics,
                     teacher_forcing=False, prefix_frames=0, initial_frame=None, **kwargs):
        if prefix_frames:
            raise ValueError("DreamX image positions belong to the first generated block")
        return _camera_tokens(tokens, grid_sizes, viewmats, intrinsics, teacher_forcing=teacher_forcing)

    def _forward_inference(self, x, t, *, initial_frame=None, current_start=0,
                           commit_cache=True, **kwargs):
        if initial_frame is not None and current_start == 0:
            x = torch.cat((initial_frame.transpose(1, 2), x[:, :, 1:]), dim=2)
            # Ignore a precomputed noisy embedding for the clamped image position.
            kwargs["patch_embeddings"] = None
            if not commit_cache:
                t = t.clone()
                t[:, 0] = 0
        return super()._forward_inference(x, t, current_start=current_start,
                                         commit_cache=commit_cache, **kwargs)

    def _forward_train(self, x, t, *, initial_frame=None, **kwargs):
        kwargs['aug_t'] = torch.full_like(t, self.context_timestep)
        if initial_frame is not None:
            x = torch.cat((initial_frame.transpose(1, 2), x[:, :, 1:]), dim=2)
            t = t.clone()
            t[:, 0] = 0
            kwargs["patch_embeddings"] = None
        return super()._forward_train(x, t, **kwargs)

class DreamXBidirectionalBlock(WanAttentionBlock):
    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, *, viewmats, Ks):
        m = [(u.squeeze(2)) for u in (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)]
        inputs = (self.norm1(x) * (1 + m[1]) + m[0]).to(self.self_attn.q.weight.dtype)
        residual = self.self_attn(inputs, seq_lens, grid_sizes, freqs)
        if hasattr(self, "cam_self_attn"):
            residual = residual + self.cam_self_attn(inputs, viewmats, Ks, seq_lens=seq_lens)
        x = x + residual * m[2]
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        inputs = (self.norm2(x) * (1 + m[4]) + m[3]).to(self.ffn[0].weight.dtype)
        return x + self.ffn(inputs) * m[5]


class DreamXBidirectionalModel(DreamXConfiguration, Wan22Model):
    block_type = DreamXBidirectionalBlock
    _no_split_modules = ["DreamXBidirectionalBlock"]

    def init_conditioning(self):
        _add_camera_branches(self, self.conditioning_config, ar=False)

    def block_kwargs(self, tokens, grid_sizes, *, viewmats, Ks):
        return _camera_tokens(tokens, grid_sizes, viewmats, Ks)


class DreamXAR(Action2VDMDGenerator, WanAR):
    """Released causal architecture; shared WanAR owns training and rollout."""
    camera_modules = ("cam_self_attn",)
    output_includes_condition = True
    inference_frames_include_condition = True
    base = "wan2.2"
    camera_control = True
    required_data_fields = ("viewmats", "intrinsics")
    inference_modes = ("i2v",)
    num_frame_per_block = 3
    local_attn_size = 12
    sink_size = 3
    image_condition_sink = False
    prefill_condition = False
    default_latent_frames = 252
    latent_shape = (48, 44, 80)
    pixel_shape = (704, 1280)
    camera_reference = "previous_block"
    # Released inference_ar_forcing.sh uses 1.0; the bare Python CLI uses 0.3.
    color_correction_strength = 1.0
    player_translation_step = .225
    player_rotation_degrees = 4.5
    player_frame_stride = 4
    player_first_frame_stride = 1
    player_pitch_sign = -1
    initial_noise_full_sequence = True

    @staticmethod
    def transformer_class():
        return DreamXCausalModel

    def add_inference_arguments(self, parser):
        super().add_inference_arguments(parser)
        parser.set_defaults(fps=16, seed=42)

    def interactive_input_options(self):
        return dict(resize_mode='resize', resize_resample='bilinear',
                    intrinsics=NORMALIZED_K)


class DreamXBidirectional(WanFullContext):
    camera_modules = ("cam_self_attn",)
    preset = "wan2.2/ti2v-5B"
    policy_class = Wan22TI2V
    initial_noise_device = "cpu"
    camera_control = True
    required_data_fields = ("viewmats", "intrinsics")
    latent_shape = (48, 44, 80)
    pixel_shape = (704, 1280)

    def __init__(self):
        super().__init__()
        self.config.sample_solver = "euler"
        # Flow Euler takes shift=5 from wan_ti2v_5b.yaml, not the CLI's --shift=3
        # (the latter is passed only to UniPC/DPM++ in the official pipeline).
        self.config.sample_guide_scale = 3.0
        self.config.sample_fps = 16
        self.config.frame_num = 81

    @staticmethod
    def transformer_class():
        return DreamXBidirectionalModel

    def load(self, model_dir, checkpoint, device, dtype):
        return WanBackend.load(self, model_dir, checkpoint, device, dtype)

    def model_inputs(self, samples, lengths, device):
        views, ks = [], []
        for row, length in zip(samples, lengths, strict=True):
            pose = torch.cat((row["condition_viewmats"], row["viewmats"]))[:length].to(device, torch.float32)
            intrinsic = torch.cat((row["condition_intrinsics"], row["intrinsics"]))[:length].to(device, torch.bfloat16)
            if len(pose) != length:
                raise ValueError("DreamX Cam camera timeline is shorter than the requested output")
            padding = max(lengths) - length
            with torch.autocast(device_type=pose.device.type, enabled=False):
                c2w = pose[:1] @ torch.linalg.inv(pose)
                c2w[0] = torch.eye(4, device=device)
                pose = _invert_SE3(c2w).to(torch.bfloat16)
            views.append(torch.cat((pose, pose[-1:].expand(padding, -1, -1))))
            ks.append(torch.cat((intrinsic, intrinsic[-1:].expand(padding, -1, -1))))
        return {"viewmats": torch.stack(views), "Ks": torch.stack(ks)}

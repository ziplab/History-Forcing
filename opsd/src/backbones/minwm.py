"""minWM Action2V final-DMD and multi-step AR backends."""

from pathlib import Path

import torch
from torch import nn

from wan.attention import attention
from wan.causal import (
    CausalWanAttentionBlock,
    CausalWanModel,
    CausalWanSelfAttention,
)
from wan.conditioning import TemporalInputs, batch_inputs
from wan.geometry import projective_qkv, relative_viewmats
from wan.kv_cache.causal import read_attention
from wan.layers import (WanAttentionBlock, WanSelfAttention,
                        rope_apply)
from wan.wan21.model import WanModel as Wan21Model

from .wan import Wan21T2V
from .wan_ar import Action2VDMDGenerator, WanAR


class MinWMSelfAttention(CausalWanSelfAttention):
    """Add minWM's projective attention to the shared RoPE result."""

    def condition_attention(
        self, query, key, value, *, block_mask, use_cache,
        current_start, tokens_per_frame, extra_sink_size, commit_cache,
        viewmats=None, Ks=None, camera_kv_cache=None,
    ):
        cache = camera_kv_cache
        if viewmats is None:
            if Ks is not None or cache is not None:
                raise ValueError("PRoPE cache and intrinsics require camera poses")
            return None
        if Ks is None:
            raise ValueError("PRoPE requires camera intrinsics")
        query, key, value, apply_output = projective_qkv(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            viewmats=viewmats, Ks=Ks,
        )
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        if not use_cache:
            output = self._padded_flex_attention(query, key, value, block_mask)
        else:
            if cache is None:
                raise ValueError("PRoPE inference requires its own KV cache")
            sink_tokens = cache.get('sink_tokens', (self.sink_size + extra_sink_size) * tokens_per_frame)
            attention_key, attention_value, key_lengths = read_attention(
                cache, key, value, current_start, sink_tokens, tokens_per_frame, commit=commit_cache)
            output = attention(query, attention_key, attention_value, k_lens=key_lengths)
        output = apply_output(output.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return self.prope_o(output.flatten(2))


class MinWMAttentionBlock(CausalWanAttentionBlock):
    attention_type = MinWMSelfAttention


class MinWMWanModel(CausalWanModel):
    block_type = MinWMAttentionBlock
    _no_split_modules = ["MinWMAttentionBlock"]
    cache_streams = ('kv_cache', 'camera_kv_cache')

    def condition_kwargs(self, inputs, batch, frames, *, prefix=False):
        views, ks = inputs.get('viewmats'), inputs.get('intrinsics')
        if views is None or views.shape != (batch, frames, 4, 4):
            raise ValueError('MinWM requires viewmats [B,F,4,4]')
        if ks is None or ks.shape != (batch, frames, 3, 3):
            raise ValueError('MinWM requires intrinsics [B,F,3,3]')
        return {('prefix_ks' if prefix else 'Ks') if name == 'intrinsics'
                else (f'prefix_{name}' if prefix else name): value for name, value in inputs.items()}

    def block_kwargs(self, tokens, grid_sizes, *, prefix_frames=0, teacher_forcing=False,
                     viewmats=None, Ks=None, prefix_viewmats=None, prefix_ks=None, **conditioning):
        inputs = super().block_kwargs(tokens, grid_sizes, prefix_frames=prefix_frames,
                                      teacher_forcing=teacher_forcing, **conditioning)
        if not prefix_frames and (prefix_viewmats is not None or prefix_ks is not None):
            raise ValueError("Condition cameras require condition embeddings")
        if prefix_frames:
            if viewmats is None:
                if prefix_viewmats is not None or prefix_ks is not None:
                    raise ValueError("Camera-free targets cannot use condition cameras")
            elif prefix_viewmats is None or prefix_ks is None:
                raise ValueError("Camera-conditioned targets require condition cameras")
        target_views, target_ks = self._expand_cameras(
            viewmats, Ks, grid_sizes, repeat=2 if teacher_forcing else 1)
        if prefix_frames and target_views is not None:
            prefix_grid = grid_sizes.clone()
            prefix_grid[:, 0] = prefix_frames
            prefix_views, prefix_ks = self._expand_cameras(
                prefix_viewmats, prefix_ks, prefix_grid, repeat=1)
            target_views = torch.cat((prefix_views, target_views), dim=1)
            target_ks = torch.cat((prefix_ks, target_ks), dim=1)
        return {**inputs, 'viewmats': target_views, 'Ks': target_ks}

    @staticmethod
    def _expand_cameras(viewmats, ks, grid_sizes, repeat, *, seq_len=None):
        if viewmats is None:
            if ks is not None:
                raise ValueError("Intrinsics require camera extrinsics")
            return None, None
        if ks is None:
            raise ValueError("Camera extrinsics require intrinsics")
        batch = len(grid_sizes)
        if viewmats.ndim != 4 or viewmats.shape[0] != batch or (
            viewmats.shape[-2:] != (4, 4)
        ):
            raise ValueError("Viewmats must have shape [B,F,4,4]")
        if ks.ndim != 4 or ks.shape[0] != batch or ks.shape[-2:] != (3, 3):
            raise ValueError("Intrinsics must have shape [B,F,3,3]")
        if repeat not in (1, 2):
            raise ValueError("Camera expansion repeat must be 1 or 2")
        expanded = ([], [])
        for sample, (frames, height, width) in enumerate(grid_sizes.tolist()):
            if (viewmats.shape[1] < frames or ks.shape[1] < frames or
                    (seq_len is None and (viewmats.shape[1] != frames or ks.shape[1] != frames))):
                raise ValueError("Camera timeline and latent timeline differ")
            count = frames * height * width
            padding = 0 if seq_len is None else seq_len - count
            if padding < 0:
                raise ValueError("Camera token count exceeds the padded sequence length")
            for source, values in zip((viewmats, ks), expanded, strict=True):
                size = source.shape[-1]
                value = source[sample, :frames, None, None].expand(
                    -1, height, width, -1, -1).reshape(count, size, size)
                if padding:
                    identity = torch.eye(size, device=value.device, dtype=value.dtype)
                    value = torch.cat((value, identity.expand(padding, -1, -1)))
                values.append(value)
        viewmats, ks = (torch.stack(values) for values in expanded)
        if repeat == 2:
            viewmats = torch.cat((viewmats, viewmats), dim=1)
            ks = torch.cat((ks, ks), dim=1)
        return viewmats, ks

    def _init_conditioning(self, control_type):
        super()._init_conditioning(control_type)
        for block in self.blocks:
            projection = nn.Linear(self.dim, self.dim).to(block.self_attn.o.weight)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            block.self_attn.prope_o = projection


class MinWMBidirectionalSelfAttention(WanSelfAttention):
    """The released bidirectional Action2V PRoPE attention."""

    def forward(self, x, seq_lens, grid_sizes, freqs, *, viewmats, Ks):
        query, key, value = self.project_qkv(x)
        rope = attention(
            rope_apply(query, grid_sizes, freqs),
            rope_apply(key, grid_sizes, freqs), value,
            k_lens=seq_lens, window_size=self.window_size,
        )
        projective_query, projective_key, projective_value, correct = (
            projective_qkv(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                viewmats=viewmats, Ks=Ks,
            )
        )
        projective = attention(
            projective_query.permute(0, 2, 1, 3),
            projective_key.permute(0, 2, 1, 3),
            projective_value.permute(0, 2, 1, 3),
            k_lens=seq_lens, window_size=self.window_size,
        )
        projective = correct(
            projective.permute(0, 2, 1, 3),
        ).permute(0, 2, 1, 3)
        return self.o(rope.flatten(2)) + self.prope_o(projective.flatten(2))


class MinWMBidirectionalBlock(WanAttentionBlock):
    self_attention_type = MinWMBidirectionalSelfAttention

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context,
                context_lens, *, viewmats, Ks):
        modulation = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        y = self.self_attn(
            self.norm1(x) * (1 + modulation[1].squeeze(2))
            + modulation[0].squeeze(2),
            seq_lens, grid_sizes, freqs, viewmats=viewmats, Ks=Ks,
        )
        x = x + y * modulation[2].squeeze(2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            self.norm2(x) * (1 + modulation[4].squeeze(2))
            + modulation[3].squeeze(2)
        )
        return x + y * modulation[5].squeeze(2)


class MinWMBidirectionalModel(Wan21Model):
    """Full-context Action2V architecture used by bidirectional checkpoints."""

    block_type = MinWMBidirectionalBlock
    _no_split_modules = ["MinWMBidirectionalBlock"]

    def init_conditioning(self):
        super().init_conditioning()
        for block in self.blocks:
            projection = nn.Linear(self.dim, self.dim).to(block.self_attn.o.weight)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            block.self_attn.prope_o = projection

    def block_kwargs(self, tokens, grid_sizes, *, viewmats, Ks, **conditioning):
        if conditioning:
            raise ValueError(f"Unsupported minWM conditions: {tuple(conditioning)}")
        viewmats, Ks = MinWMWanModel._expand_cameras(
            viewmats, Ks, grid_sizes, repeat=1, seq_len=tokens.shape[1],
        )
        return {"viewmats": viewmats, "Ks": Ks}


class MinWMBidirectional(Wan21T2V):
    """Reusable full-context minWM Action2V backend."""

    camera_modules = ("self_attn.prope_o",)
    required_data_fields = ("viewmats", "intrinsics")
    camera_control = True

    @staticmethod
    def checkpoint_file(checkpoint):
        path = Path(checkpoint)
        model = path / "model.pt"
        return model if path.is_dir() and model.is_file() else Wan21T2V.checkpoint_file(path)

    @staticmethod
    def transformer_class():
        return MinWMBidirectionalModel

    def model_inputs(self, samples, lengths, device):
        inputs = []
        for sample, length in zip(samples, lengths, strict=True):
            prefix = len(sample["condition_latent"])
            values = {}
            for name in ("viewmats", "intrinsics"):
                if len(sample[name]) < length or len(sample[f"condition_{name}"]) != prefix:
                    raise ValueError("Camera inputs do not cover the prefix and requested target")
                value = torch.cat((sample[f"condition_{name}"], sample[name][:length])).to(device)
                if name == "viewmats":
                    value = relative_viewmats(value.float())
                values["Ks" if name == "intrinsics" else name] = value[prefix:].to(torch.bfloat16)[None]
            inputs.append(TemporalInputs(values).pad(max(lengths)))
        return batch_inputs(inputs).forward_kwargs()


class MinWMAR(WanAR):
    """minWM teacher-forced AR diffusion with 50-step UniPC CFG."""

    supports_sft = True
    camera_modules = ("self_attn.prope_o",)
    base = "wan2.1"
    num_frame_per_block = 4
    local_attn_size = 20
    sink_size = 0
    camera_control = True
    default_latent_frames = 20
    sampling_mode = "ode"
    sampling_steps = 50
    sampling_steps_type = int
    guidance_scale = 3.0

    def training_inputs(self, batch, device):
        target, prefix, ks, prefix_ks = self.camera_batch(batch, device)
        return (TemporalInputs({'viewmats': prefix, 'intrinsics': prefix_ks}),
                TemporalInputs({'viewmats': target, 'intrinsics': ks}))

    @staticmethod
    def transformer_class():
        return MinWMWanModel

    def camera_batch(self, batch, device):
        """Prepare valid camera rows once; retain the collator's unused padding."""
        if not self.camera_control:
            return None, None, None, None
        target = batch["viewmats"].to(device=device, dtype=torch.float32, non_blocking=True).clone()
        condition = batch["condition_viewmats"].to(device=device, dtype=torch.float32, non_blocking=True).clone()
        intrinsics = batch["intrinsics"].to(device=device, dtype=torch.float32, non_blocking=True).clone()
        condition_intrinsics = batch["condition_intrinsics"].to(device=device, dtype=torch.float32, non_blocking=True).clone()
        for row, (prefix, length) in enumerate(zip(
            batch["condition_mask"].sum(1).tolist(), batch["valid_frame_mask"].sum(1).tolist(), strict=True,
        )):
            sample = {
                "viewmats": target[row, :length], "condition_viewmats": condition[row, :prefix],
                "intrinsics": intrinsics[row, :length],
                "condition_intrinsics": condition_intrinsics[row, :prefix],
            }
            inputs = self.prepare_inputs(sample, length, device, dtype=torch.float32).temporal
            condition[row, :prefix] = inputs["viewmats"][0, :prefix]
            target[row, :length] = inputs["viewmats"][0, prefix:]
            condition_intrinsics[row, :prefix] = inputs["intrinsics"][0, :prefix]
            intrinsics[row, :length] = inputs["intrinsics"][0, prefix:]
        return tuple(value.to(torch.bfloat16) for value in (target, condition, intrinsics, condition_intrinsics))

    def add_inference_arguments(self, parser):
        self.add_common_inference_arguments(parser)
        parser.add_argument(
            "--sampling-steps", type=int, default=None
        )
        parser.add_argument(
            "--guidance-scale", type=float, default=None
        )
        parser.add_argument(
            "--negative-prompt", default=None
        )

    def inference_options(self, args):
        options = self.common_inference_options(args)
        steps = self.sampling_steps if args.sampling_steps is None else args.sampling_steps
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValueError("sampling_steps must be an integer in [1,1000]")
        return {
            **options,
            "sampling_steps": steps,
            **self.inference_text_options({**options, 'guidance_scale': args.guidance_scale,
                                           'negative_prompt': args.negative_prompt}),
        }


class MinWM(Action2VDMDGenerator, MinWMAR):
    """minWM Action2V few-step checkpoint."""

    supports_sft = False
    sampling_mode = "sde"
    sampling_steps = (1000, 750, 500, 250)
    sampling_steps_type = staticmethod(WanAR.sampling_steps_type)
    guidance_scale = 1.0
    add_inference_arguments = WanAR.add_inference_arguments
    inference_options = WanAR.inference_options

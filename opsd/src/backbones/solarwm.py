# Copyright 2026 SolarWM Contributors. Apache-2.0.
"""Local SolarWM TI2V-5B Stage 1 AnyFlow and final Stage 2 SGF models.

Localized from SolarWM runtime/modeling/causal_model.py, inference.py,
and stage2.py. Wan parameters and geometry reuse the shared local
implementation.
"""

from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from tqdm import trange

from wan.geometry import _invert_SE3, _apply_tiled_projmat, projective_matrices, projective_qkv
from wan.layers import Head, WanAttentionBlock, WanSelfAttention
from wan.layers import rope_apply, embed_time
from wan.wan22.model import WanModel
from model import WanModelWrapper
from scheduler import DEFAULT_FEW_STEP_TIMESTEPS, make_solar_sgf_scheduler, sample_block_noises, sampling_step
from .wan_ar import WanAR


class SolarSelfAttention(WanSelfAttention):
    def forward(self, x, grid_sizes, freqs, camera, block_mask, kv_cache=None,
                current_start=0, commit_cache=False, cache_geometry=None):
        q, k, v = self.project_qkv(x)
        if kv_cache is None:
            q = rope_apply(q, grid_sizes, freqs).type_as(v)
            k = rope_apply(k, grid_sizes, freqs).type_as(v)
            q, k, v, apply_output = projective_qkv(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                viewmats=camera["viewmats"], Ks=camera["K"],
            )
            x = flex_attention(q, k, v, block_mask=block_mask)
        else:
            from wan.causal import causal_rope_apply
            from wan.attention import attention
            from wan.kv_cache.causal import read_attention
            from wan.kv_cache.ops import _rope_time_delta_mul_
            projection, inverse, origin = cache_geometry
            tokens = int(grid_sizes[0, 1] * grid_sizes[0, 2])
            q = causal_rope_apply(q, grid_sizes, freqs, current_start // tokens - origin).type_as(v)
            # Keep one canonical pre-PRoPE K/V copy. RoPE origins and cameras
            # change only in the attention view, never in the committed bank.
            k = causal_rope_apply(k, grid_sizes, freqs, current_start // tokens).type_as(v)
            k, v, lengths, positions = read_attention(
                kv_cache, k, v, current_start, 0, tokens, commit=commit_cache,
                freqs=freqs, return_positions=True)
            frame_ids = positions // tokens
            if origin:
                k = k.clone()
                _rope_time_delta_mul_(k, freqs, -origin)
            matrices = inverse.gather(1, frame_ids[:, :, None, None].expand(-1, -1, 4, 4))
            q = _apply_tiled_projmat(q.transpose(1, 2), projection.transpose(-1, -2).to(inverse.dtype))
            k = _apply_tiled_projmat(k.transpose(1, 2), matrices).transpose(1, 2)
            v = _apply_tiled_projmat(v.transpose(1, 2), matrices).transpose(1, 2)
            apply_output = partial(_apply_tiled_projmat, matrix=projection)
            x = attention(q.transpose(1, 2), k, v, k_lens=lengths).transpose(1, 2)
        return self.o(apply_output(x).transpose(1, 2).flatten(2))


class SolarAttentionBlock(WanAttentionBlock):
    self_attention_type = SolarSelfAttention

    def forward(self, x, e, grid_sizes, freqs, context, camera, block_mask,
                kv_cache=None, current_start=0, commit_cache=False, cache_geometry=None):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            grid_sizes, freqs, camera, block_mask, kv_cache, current_start, commit_cache, cache_geometry,
        )
        x = x + y * e[2].squeeze(2)
        x = x + self.cross_attn(self.norm3(x), context, None)
        y = self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        return x + y * e[5].squeeze(2)


class SolarHead(Head):
    def forward(self, x, e):
        e = (self.modulation.unsqueeze(1) + e.unsqueeze(2)).chunk(2, dim=2)
        return self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))


class SolarWanModel(WanModel):
    """Wan2.2 with fused PRoPE and causal KV attention."""

    block_type = SolarAttentionBlock
    head_type = SolarHead
    _no_split_modules = ["SolarAttentionBlock"]

    def __init__(self, *, flow_objective="anyflow_forward_map", **config):
        super().__init__(**config)
        if flow_objective not in ("flow_matching", "anyflow_forward_map"):
            raise ValueError(f"Unsupported SolarWM flow objective: {flow_objective}")
        self.flow_objective = flow_objective
        if self.uses_anyflow:
            self.delta_embedding = deepcopy(self.time_embedding)
            self.register_buffer("anyflow_gate_buffer", torch.tensor([0.25]), persistent=False)
        self._window_masks = {}

    @property
    def uses_anyflow(self):
        return self.flow_objective == "anyflow_forward_map"

    def time_condition(self, t, r=None):
        time = embed_time(self.time_embedding, t, self.freq_dim).flatten(0, 1)
        if not self.uses_anyflow:
            if r is not None:
                raise ValueError("SolarWM flow-matching forward does not accept r")
            return time
        if r is None or r.shape != t.shape:
            raise ValueError("SolarWM AnyFlow requires r aligned with t")
        delta = embed_time(self.delta_embedding, r, self.freq_dim).flatten(0, 1)
        gate = self.anyflow_gate_buffer.to(device=time.device, dtype=time.dtype)
        return (1.0 - gate) * time + gate * delta

    def window_mask(self, frames, tokens, chunk, device):
        key = (frames, tokens, chunk, str(device))
        if key not in self._window_masks:
            total, block = frames * tokens, chunk * tokens
            padded = (total + 127) // 128 * 128

            def mask(b, h, q, k):
                end = torch.clamp((q // block + 1) * block, max=total)
                return (q == k) | ((q < total) & (k < end))

            # Same full/partial 128-token blocks as upstream, O(blocks^2),
            # without materializing a quadratic token mask on the GPU.
            starts = torch.arange(padded // 128) * 128
            ends = starts + 128
            first_end = ((starts // block + 1) * block).clamp(max=total)
            last_end = (((ends.clamp(max=total) - 1) // block + 1) * block).clamp(max=total)
            full = (ends[:, None] <= total) & (ends[None, :] <= first_end[:, None])
            any_hit = ((starts[:, None] < total) & (starts[None, :] < last_end[:, None]))
            any_hit |= torch.eye(len(starts), dtype=torch.bool)

            def metadata(dense):
                dense = dense.int()
                counts = dense.sum(-1).int()[None, None].to(device)
                indices = dense.argsort(dim=-1, descending=True, stable=True).int()[None, None].to(device)
                return counts, indices

            self._window_masks[key] = BlockMask.from_kv_blocks(
                *metadata(any_hit & ~full), *metadata(full), BLOCK_SIZE=128, mask_mod=mask,
            )
        return self._window_masks[key]

    def forward(self, x, t, context, seq_len, *, r=None, y_camera, clean_history=None,
                num_frame_per_block=3, kv_cache=None, current_start=0, commit_cache=False,
                rope_origin=0):
        if x.ndim != 5 or t.shape != (x.shape[0], seq_len):
            raise ValueError("SolarWM needs BCFHW input and aligned [B,tokens] t")
        if self.uses_anyflow and (r is None or r.shape != t.shape):
            raise ValueError("SolarWM AnyFlow needs [B,tokens] r aligned with t")
        if not self.uses_anyflow and r is not None:
            raise ValueError("SolarWM Stage 2 does not use AnyFlow r conditioning")
        if any(size % patch for size, patch in zip(x.shape[2:], self.patch_size)):
            raise ValueError("SolarWM latent dimensions must align with the patch grid")
        self.freqs = self.freqs.to(x.device)
        noisy_pe = self.embed_video(x)
        grids = torch.tensor([p.shape[2:] for p in noisy_pe], dtype=torch.long)
        noisy = torch.cat([p.flatten(2).transpose(1, 2) for p in noisy_pe])
        if noisy.shape[1] != seq_len:
            raise ValueError("SolarWM seq_len must equal the current block token count")
        history_frames = 0 if clean_history is None else clean_history.shape[2]
        if kv_cache is not None and history_frames:
            raise ValueError("Cached SolarWM reads committed KV, not clean_history")
        if history_frames:
            clean = torch.cat([p.flatten(2).transpose(1, 2) for p in self.embed_video(clean_history)])
            full = torch.cat((clean, noisy), dim=1)
            zeros = t.new_zeros(t.shape[0], clean.shape[1])
            time = torch.cat((zeros, t), 1)
            target = torch.cat((zeros, r), 1) if self.uses_anyflow else None
        else:
            full, time, target = noisy, t, r
        frames, height, width = grids[0].tolist()
        tokens = height * width
        frames += history_frames
        if kv_cache is None and frames % num_frame_per_block:
            raise ValueError("SolarWM history and current chunk must contain complete blocks")
        e = self.time_condition(time, target)
        e0 = self.time_projection(e.to(self.time_projection[1].weight.dtype)).float().unflatten(1, (6, self.dim)).unflatten(0, time.shape)
        head_e = self.time_condition(t, r).unflatten(0, t.shape)
        context = self.project_context(context)
        block_mask = (self.window_mask(frames, tokens, num_frame_per_block, x.device)
                      if kv_cache is None else None)
        full_grid = grids.clone()
        full_grid[:, 0] = frames
        camera = cache_geometry = None
        if kv_cache is None:
            if (y_camera["viewmats"].shape != (x.shape[0], frames * tokens, 4, 4)
                    or y_camera["K"].shape != (x.shape[0], frames * tokens, 3, 3)):
                raise ValueError("SolarWM SFT cameras must cover every active window token")
            padding = (-full.shape[1]) % 128
            full = F.pad(full, (0, 0, 0, padding))
            e0 = F.pad(e0, (0, 0, 0, 0, 0, padding))
            camera = {name: torch.cat((value, torch.eye(value.shape[-1], device=x.device, dtype=value.dtype)
                                      [None, None].expand(x.shape[0], padding, -1, -1)), dim=1)
                      for name, value in y_camera.items()}
        else:
            # Camera matrices are shared by all layers; cached frame IDs
            # select the appropriate inverse independently in each layer.
            projection, inverse = projective_matrices(y_camera['viewmats'], y_camera['K'])
            start = current_start // tokens
            projection = projection[:, start:start+frames].repeat_interleave(tokens, dim=1)
            cache_geometry = projection, inverse, rope_origin
        for index, block in enumerate(self.blocks):
            call = partial(block, e=e0, grid_sizes=full_grid, freqs=self.freqs,
                           context=context, camera=camera, block_mask=block_mask,
                           kv_cache=None if kv_cache is None else kv_cache[index],
                           current_start=current_start, commit_cache=commit_cache,
                           cache_geometry=cache_geometry)
            full = (self._gradient_checkpointing_func(call, full)
                    if torch.is_grad_enabled() and self.gradient_checkpointing else call(full))
        noisy_output = full[:, history_frames * tokens:frames * tokens]
        return torch.stack(self.unpatchify(self.head(noisy_output, head_e), grids))




class SolarWM(WanAR):
    """Released SolarWM-5B-tf-stage1-81f with adjacent flow maps."""

    inference_modes = ("i2v",)
    output_includes_condition = True
    supports_sft = True
    supports_opsd = True
    base = "wan2.2/ti2v-5B"
    local_attn_size = 18
    sink_size = 0
    image_condition_sink = False
    prefill_condition = False
    camera_control = True
    player_action_full_block = True
    required_data_fields = ("viewmats", "intrinsics")
    pixel_shape = (480, 864)
    latent_shape = (48, 30, 54)
    num_frame_per_block = 3
    default_num_latent_frames = 60
    sampling_steps = 50
    sampling_mode = "ode"
    sampling_steps_type = int
    flow_objective = "anyflow_forward_map"
    checkpoint_stage = "stage1"
    checkpoint_objective = "anyflow_forward_map"
    release_name = "Stage 1"

    @staticmethod
    def transformer_class():
        return SolarWanModel

    def build(self, model_dir, config):
        self.validate_model_config(config)
        from wan.configs import get_config, model_config

        return WanModelWrapper(self.transformer_class()(
            flow_objective=self.flow_objective,
            **model_config(get_config("wan2.2/ti2v-5B")),
        ))

    @staticmethod
    def checkpoint_file(checkpoint):
        checkpoint = Path(checkpoint)
        return checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint

    def checkpoint_config(self, checkpoint):
        payload = torch.load(self.checkpoint_file(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        return payload["config"] if "model" in payload else None

    def load_weights(self, model, checkpoint, *, assign=False):
        payload = torch.load(
            self.checkpoint_file(checkpoint), map_location="cpu", weights_only=True, mmap=True,
        )
        if "model" in payload:
            saved = payload["config"]
            self.validate_model_config(saved)
            return self.unwrap(model).load_state_dict(payload["model"], strict=True, assign=assign)
        saved = payload.get("config")
        if not isinstance(saved, dict):
            raise ValueError("SolarWM checkpoint has no configuration mapping")
        # The released export saves a semantic contract, not the inference YAML.
        # Match upstream _validate_checkpoint_config; block/window/t/r settings
        # belong to this backend's released Stage 1 runtime, not checkpoint fields.
        expected = {
            "model": {"family": "wan22_ti2v_5b", "camera_translation_transform": "linear"},
            "train": {"stage": self.checkpoint_stage, "objective": self.checkpoint_objective},
        }
        for section, values in expected.items():
            for key, value in values.items():
                default = "linear" if key == "camera_translation_transform" else None
                actual = saved.get(section, {}).get(key, default)
                if str(actual) != value:
                    raise ValueError(
                        f"SolarWM {self.release_name} requires {section}.{key}={value!r}; "
                        f"got {actual!r}"
                    )
        self.validate_official_checkpoint(payload, saved)
        weight_key = self.official_weight_key(payload)
        state = payload[weight_key]
        prefixed = [key.startswith("model.") for key in state]
        if not state or (any(prefixed) and not all(prefixed)):
            raise ValueError("SolarWM generator keys must be nonempty and uniformly model-prefixed or unprefixed")
        state = {key.removeprefix("model."): value for key, value in state.items()}
        print(f"[SolarWM] loading {weight_key}: {checkpoint}", flush=True)
        return self.unwrap(model).model.load_state_dict(state, strict=True, assign=assign)

    def validate_official_checkpoint(self, payload, config):
        del payload, config

    @staticmethod
    def official_weight_key(payload):
        # Prefer the stage-1 generator weights when both variants are available.
        return "generator" if "generator" in payload else "generator_ema"

    def load(self, model_dir, checkpoint, device, dtype):
        from accelerate import init_empty_weights

        # Empty construction avoids allocating another CPU copy of the 5B DiT.
        with init_empty_weights():
            stored = self.checkpoint_config(checkpoint)
            model = self.build(
                model_dir, self.model_config(None) if stored is None else stored,
            )
        self.load_weights(model, checkpoint, assign=True)
        # Convert mapped CPU weights before transfer; keep no FP32 DiT on CUDA.
        return model.to(dtype=dtype).to(device=device).eval().requires_grad_(False)

    def codec_config(self, model_dir):
        from wan.configs import codec_config

        config = codec_config(48)
        config["vae_checkpoint"] = "vae/Wan2.2_VAE.pth"
        return config

    def text_encoder(self, model_dir, device, *, shard_fn=None):
        from wan.t5 import T5EncoderModel

        model_dir = Path(model_dir)
        return T5EncoderModel(
            text_len=512, device=device,
            checkpoint_path=str(model_dir / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(model_dir / "tokenizer"), shard_fn=shard_fn,
        )

    def validate(self, model_dir, checkpoint, *, codec=False, config=None):
        model_dir = Path(model_dir)
        self.validate_model_config(self.model_config(None) if config is None else config)
        paths = [self.checkpoint_file(checkpoint),
                 model_dir / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth"]
        if codec:
            paths.append(model_dir / self.codec_config(model_dir)["vae_checkpoint"])
        missing = [str(path) for path in paths if not path.is_file()]
        if not (model_dir / "tokenizer").is_dir():
            missing.append(str(model_dir / "tokenizer"))
        if missing:
            raise FileNotFoundError("Missing SolarWM assets:\n" + "\n".join(missing))
        self.transformer_class()

    def model_config(self, args):
        return {}

    def validate_model_config(self, config):
        if not isinstance(config, dict) or config:
            raise ValueError("SolarWM Stage 1 model config must be an empty mapping")

    @property
    def uses_anyflow(self):
        return self.flow_objective == "anyflow_forward_map"


    def schedule(self, device, config=None):
        from scheduler import make_flow_training_schedule
        return make_flow_training_schedule(device, self.timestep_shift)

    def flow_forward(self, model, batch, context, device, schedule, **options):
        """Teacher-forced flow matching with the image excluded from loss."""
        from scheduler import sample_flow_training
        from wan.conditioning import TemporalInputs
        batch = self.trim_training_targets(batch)
        condition, condition_frames = self.training_condition(batch, device)
        self.validate_training_lengths([condition_frames], [batch['clean_latent'].shape[1]])
        clean = self._move(batch['clean_latent'], device)
        full = torch.cat((condition, clean), dim=1)
        prefix, target_inputs = self.training_inputs(batch, device)
        inputs = TemporalInputs({name: torch.cat((prefix.temporal[name], value), dim=1)
                                 for name, value in target_inputs.temporal.items()},
                                {'first_latent': condition})
        noisy, target, times, weight = sample_flow_training(full, (3,) * (full.shape[1] // 3), schedule)
        predictions = []
        secondary = options.get('secondary_context')
        prompt_mask = batch.get('teacher_prompt_mask')
        if (secondary is None) != (prompt_mask is None):
            raise ValueError('Teacher context and prompt mask must be provided together')
        if prompt_mask is not None:
            prompt_mask = torch.cat((prompt_mask[:, :1], prompt_mask), dim=1)
        for start in range(0, full.shape[1], 3):
            block_context = context
            if prompt_mask is not None:
                flags = prompt_mask[:, start:start+3]
                if not torch.all(flags == flags[:, :1]):
                    raise ValueError('SolarWM prompt changes must align with AR block boundaries')
                block_context = [secondary[i] if use_teacher else context[i]
                                 for i, use_teacher in enumerate(flags[:, 0].tolist())]
            history_start = max(0, start + 3 - self.local_attn_size)
            predictions.append(self.predict_block(
                model, noisy[:, start:start+3], block_context, inputs, start,
                times[:, start:start+3],
                times[:, start:start+3] if self.uses_anyflow else None,
                cache=None,
                clean_history=full[:, history_start:start],
            ))
        prediction = torch.cat(predictions, dim=1)[:, 1:]
        return self.masked_flow_output(prediction, target[:, 1:], weight[:, 1:], clean,
                                       batch['valid_frame_mask'].to(device))

    def opsd_schedule(self, device, sampling_steps, *, timestep_shift=None, sampling_mode=None):
        from scheduler import make_flow_map_scheduler
        if sampling_mode not in (None, "ode"):
            raise ValueError("SolarWM AnyFlow requires deterministic endpoint updates (sampling_mode=ode)")
        self.validate_sampling_steps(sampling_steps)
        return make_flow_map_scheduler(device, sampling_steps, shift=self.timestep_shift if timestep_shift is None else timestep_shift)

    def validate_sampling_steps(self, sampling_steps):
        if type(sampling_steps) is not int or sampling_steps <= 0:
            raise ValueError("SolarWM sampling_steps must be a positive integer")

    def validate_training_lengths(self, condition_lengths, target_lengths):
        if set(condition_lengths) != {1} or min(target_lengths) < 2:
            raise ValueError("SolarWM training requires one condition frame and at least two target frames")

    def training_batch_key(self, condition_frames, target_frames):
        # SFT makes one distributed forward per block; all ranks must agree.
        return (condition_frames, self.temporal_layout.complete_target_frames(
            target_frames, prefix_frames=condition_frames))

    def prepare_inputs(self, sample, target_frames, device, *, dtype=torch.bfloat16):
        from wan.conditioning import TemporalInputs
        inputs = super().prepare_inputs(sample, target_frames, device, dtype=torch.float32)
        return TemporalInputs(inputs.temporal, {"first_latent": inputs.static['initial_frame']})

    def new_cache(self, model, inputs, frames, height, width, *, capacity_frames=None, sink_frames=None):
        from wan.kv_cache.causal import CausalCache, allocate_cache, cache_capacity_frames, resolve_sink_frames
        inner = self.unwrap(model).model
        if frames > len(inner.freqs):
            raise ValueError(f"SolarWM KV cache supports at most {len(inner.freqs)} latent frames")
        first = inputs.static['first_latent']
        tokens = height * width // 4
        sink = resolve_sink_frames(sink_frames, 0)
        capacity = cache_capacity_frames(frames, self.local_attn_size, sink, block_frames=3)
        if capacity_frames is not None:
            capacity = capacity_frames
        cache = self.unwrap(model).guided_cache(lambda: CausalCache({'kv_cache': allocate_cache(
            inner, first.shape[0], capacity * tokens, first.device, first.dtype,
            sink_tokens=sink * tokens)}, []))
        return cache

    def predict_block(self, model, latent, context, inputs, start, t, r,
                      *, cache, clean_history=None, commit=False):
        """Read causal KV; only SFT supplies differentiable clean history."""
        batch, frames, _, height, width = latent.shape
        tokens = height * width // 4
        times = torch.as_tensor(t, device=latent.device, dtype=torch.float32).expand(batch, frames).clone()
        targets = (torch.as_tensor(r, device=latent.device, dtype=torch.float32)
                   .expand(batch, frames).clone() if self.uses_anyflow else None)
        if start == 0:
            latent = torch.cat((inputs.static['first_latent'], latent[:, 1:]), dim=1)
            times[:, 0] = 0
            if targets is not None:
                targets[:, 0] = 0
        history_start = max(0, start + frames - self.local_attn_size)
        views, intrinsics = inputs.temporal['viewmats'], inputs.temporal['intrinsics']
        if cache is None:
            camera = self.camera_window(views, intrinsics, history_start, start + frames, tokens)
            clean = clean_history.transpose(1, 2) if start else None
            kwargs = {}
        else:
            with torch.autocast(device_type=views.device.type, enabled=False):
                relative = views @ _invert_SE3(views[:, history_start:history_start+1])
            camera = {'viewmats': relative, 'K': intrinsics}
            clean = None
            kwargs = dict(kv_cache=cache.streams['kv_cache'], current_start=start*tokens,
                          commit_cache=commit, rope_origin=history_start)
            kwargs.update({name: layers for name, layers in cache.streams.items() if name != 'kv_cache'})
        prediction = model(latent.transpose(1, 2), t=times.repeat_interleave(tokens, 1),
                           r=None if targets is None else targets.repeat_interleave(tokens, 1),
                           context=context, seq_len=frames*tokens,
                           y_camera=camera, clean_history=clean, num_frame_per_block=3, **kwargs).transpose(1, 2)
        if start == 0:
            prediction = torch.cat((torch.zeros_like(prediction[:, :1]), prediction[:, 1:]), dim=1)
        return prediction

    @torch.inference_mode(False)
    def _opsd_loss_terms(self, recipe, model, teacher, sample, contexts, schedule, seed,
                        *, mse_space='velocity', use_adapters=False, validate=False, sink_frames=None,
                        history_guidance_scale=3.0, teacher_kv=None, num_supervised_denoising_steps=None,
                        supervision_max_timestep=None):
        """Keep native rollouts; score uses (t,t), clean x0 uses (t,0)."""
        from recipes.opsd import TeacherKVConfig, prediction_mse
        recipe.validate_loss_options(mse_space, history_guidance_scale)
        supervised_steps = recipe.supervision_steps(
            schedule, num_supervised_denoising_steps, supervision_max_timestep=supervision_max_timestep,
        )
        teacher_kv = TeacherKVConfig.from_value(teacher_kv)
        recent_frames = teacher_kv.shared_recent_frames
        block_frames = max(sample['block_lengths'])
        positive, negative = teacher_kv.resolve(0, self.local_attn_size, block_frames, sample['revisit'])
        student_inputs, teacher_inputs = sample['conditioning'].student, sample['conditioning'].teacher
        first = student_inputs.static['first_latent']
        batch, _, channels, height, width = first.shape
        student_cache = self.new_cache(
            model, student_inputs, sample['target_frames'], height, width, sink_frames=sink_frames)
        teacher_cache = self.new_cache(model, teacher_inputs, sample['target_frames'], height, width)
        tokens_per_frame = height * width // 4
        teacher_revisit_cache = self.new_cache(
            model, teacher_inputs, sample['target_frames'], height, width,
            capacity_frames=positive.capacity(recent_frames, block_frames))
        recent_local_cache = (self.new_cache(
            model, teacher_inputs, sample['target_frames'], height, width,
            capacity_frames=negative.capacity(recent_frames, block_frames))
                              if history_guidance_scale != 1.0 else None)
        generators = (tuple(torch.Generator(device=first.device).manual_seed(int(s)) for s in seed)
                      if isinstance(seed, (list, tuple)) else
                      torch.Generator(device=first.device).manual_seed(int(seed)))
        stochastic = schedule.stochastic_sampling
        student_recent = first[:, :0] if not recent_frames else first
        timeline = tuple(zip(sample['block_starts'], sample['block_lengths'], strict=True))
        # Grad-enabled loss forwards reuse the native, deterministic layer mode.
        model.eval()
        for start, frames in timeline:
            latent, transitions = sample_block_noises(
                (batch, frames, channels, height, width),
                len(schedule.timesteps) if stochastic else 1,
                generators, first.device, first.dtype,
            )
            if start == 0:
                latent[:, 0] = first[:, 0]
            needs_teacher_history = start + frames < sample['target_frames']
            teacher_latent = latent.clone() if needs_teacher_history else None
            supervised = start >= sample['revisit']

            def encode_recent(cache):
                with recipe.teacher_parameters(model, teacher, use_adapters), torch.no_grad():
                    recent_start = start - recent_frames
                    for previous_start, previous_frames in timeline:
                        begin = max(previous_start, recent_start)
                        end = min(previous_start + previous_frames, start)
                        if begin < end:
                            self.predict_block(model, student_recent[:, begin-recent_start:end-recent_start],
                                               contexts[0], student_inputs, begin, 0.0,
                                               0.0 if self.uses_anyflow else None,
                                               cache=cache, commit=True)

            if supervised:
                for cache, policy in ((teacher_revisit_cache, positive), (recent_local_cache, negative)):
                    if cache is None:
                        continue
                    cache.seed_from(teacher_cache, start - recent_frames, tokens_per_frame,
                                    sink_frames=policy.sink_frames,
                                    history_frames=None if policy.history_frames == 'native' else policy.history_frames,
                                    rebase_limit=None)
                    if recent_frames:
                        encode_recent(cache)
            for index, (t, r) in enumerate(zip(schedule.timesteps, schedule.targets, strict=True)):
                if supervised and supervised_steps[index]:
                    # Average flow over (t,r) is neither the instantaneous field
                    # nor the full clean prediction; condition on the loss endpoint.
                    loss_r = ({'score': t, 'velocity': r, 'x': 0.0}[mse_space]
                              if self.uses_anyflow else None)
                    native_loss = not self.uses_anyflow or mse_space == 'velocity' or (
                        mse_space == 'x' and index == len(schedule.timesteps) - 1)
                    local_prediction = None
                    # Both teacher branches share the noisy latent, timestep,
                    # prompt, control and references; only the history differs.
                    with recipe.teacher_parameters(model, teacher, use_adapters), torch.no_grad():
                        if recent_local_cache is not None:
                            local_prediction = self.predict_block(
                                model, latent, contexts[0], student_inputs, start, t, loss_r,
                                cache=recent_local_cache)
                        positive_prediction = self.predict_block(
                            model, latent, contexts[0], student_inputs, start, t, loss_r, cache=teacher_revisit_cache)
                    target = recipe.teacher_target(positive_prediction, local_prediction, history_guidance_scale)
                    with recipe.student_parameters(model, use_adapters):
                        with torch.enable_grad():
                            prediction = self.predict_block(model, latent, contexts[0],
                                                            student_inputs, start, t, loss_r, cache=student_cache)
                            loss = prediction_mse(prediction, target.detach(), schedule.sigmas[index], mse_space)
                            if not loss.requires_grad:
                                raise RuntimeError('SolarWM OPSD student forward has no gradient graph')
                        yield loss
                        # Backward has finished before changing adapters or cached history.
                        del loss, target, positive_prediction, local_prediction
                        if native_loss:
                            native_prediction = prediction.detach()
                        del prediction
                    with recipe.student_parameters(model, use_adapters), torch.no_grad():
                        if not native_loss:
                            native_prediction = self.predict_block(model, latent, contexts[0],
                                                                   student_inputs, start, t, r, cache=student_cache)
                        latent = sampling_step(schedule,
                            native_prediction, latent, index,
                            **({'noise': transitions[index]}
                               if stochastic and index < len(transitions) else {}),
                        )
                    del native_prediction
                else:
                    native_r = r if self.uses_anyflow else None
                    with recipe.student_parameters(model, use_adapters), torch.no_grad():
                        prediction = self.predict_block(model, latent, contexts[0],
                                                        student_inputs, start, t, native_r,
                                                        cache=student_cache)
                        latent = sampling_step(schedule,
                            prediction, latent, index,
                            **({'noise': transitions[index]}
                               if stochastic and index < len(transitions) else {}),
                        )
                if start == 0:
                    latent[:, 0] = first[:, 0]
            if needs_teacher_history:
                # Advance independent fixed history after supervision/backward.
                for index, (t, r) in enumerate(zip(schedule.timesteps, schedule.targets, strict=True)):
                    with recipe.teacher_parameters(model, teacher, use_adapters), torch.no_grad():
                        prediction = self.predict_block(model, teacher_latent, contexts[1],
                                                        teacher_inputs, start, t, r if self.uses_anyflow else None,
                                                        cache=teacher_cache)
                        teacher_latent = sampling_step(schedule,
                            prediction, teacher_latent, index,
                            **({'noise': transitions[index]}
                               if stochastic and index < len(transitions) else {}),
                        )
                    if start == 0:
                        teacher_latent[:, 0] = first[:, 0]
            if needs_teacher_history:
                with recipe.teacher_parameters(model, teacher, use_adapters), torch.no_grad():
                    self.predict_block(model, teacher_latent, contexts[1], teacher_inputs,
                                       start, 0.0, 0.0 if self.uses_anyflow else None,
                                       cache=teacher_cache, commit=True)
            with recipe.student_parameters(model, use_adapters), torch.no_grad():
                self.predict_block(model, latent, contexts[0], student_inputs,
                                   start, 0.0, 0.0 if self.uses_anyflow else None,
                                   cache=student_cache, commit=True)
            if recent_frames:
                student_recent = (latent.detach() if start == 0 else
                                  torch.cat((student_recent, latent.detach()), dim=1))[:, -recent_frames:]
        if validate:
            histories = [student_cache, teacher_cache, teacher_revisit_cache]
            if recent_local_cache is not None:
                histories.append(recent_local_cache)
            recipe.validate_gradients(model, use_adapters, histories)

    def add_inference_arguments(self, parser):
        self.add_cache_arguments(parser)
        parser.add_argument("--num-latent-frames", type=int, default=self.default_num_latent_frames,
                            help="Complete output length, including the first image; multiple of 3")
        parser.add_argument('--use-data-length', action='store_true',
                            help="Generate each packed sample's complete native timeline")
        parser.add_argument("--sampling-steps", type=int, default=None,
                            help="Adjacent flow-map steps per block (official passes: 50 or 4)")
        parser.add_argument("--timestep-shift", type=float, default=None,
                            help="Override the checkpoint or native timestep shift")
        parser.add_argument("--sampling-mode", choices=("ode", "sde"), default=None)

    def inference_options(self, args):
        frames = None if args.use_data_length else args.num_latent_frames
        if frames is not None and (frames <= 0 or frames % self.num_frame_per_block):
            raise ValueError("SolarWM output length must be a positive multiple of 3")
        steps = self.sampling_steps if args.sampling_steps is None else args.sampling_steps
        if type(steps) is not int or steps <= 0:
            raise ValueError("SolarWM sampling_steps must be a positive integer")
        from scheduler import shift_sigmas
        shift = self.timestep_shift if args.timestep_shift is None else args.timestep_shift
        shift_sigmas(0.0, shift)
        sampling_mode = args.sampling_mode or self.sampling_mode
        self.opsd_schedule('cpu', steps, timestep_shift=shift,
                           sampling_mode=sampling_mode)
        return dict(num_latent_frames=frames, sampling_steps=steps,
                    timestep_shift=shift,  sampling_mode=sampling_mode,
                    sink_frames=getattr(args, 'sink_frames', None))

    def prepare_interactive_inputs(self, sample, target_frames, device, dtype):
        from wan.conditioning import TemporalInputs
        return TemporalInputs(
            {name: torch.cat((sample['condition_' + name], sample[name][:target_frames]))
             .to(device=device, dtype=torch.float32)[None] for name in ('viewmats', 'intrinsics')},
            {'first_latent': sample['condition_latent'][None].to(device=device, dtype=dtype)},
        )

    def update_interactive_camera(self, inputs, sample, start, length, device, dtype):
        # SolarWM rebases global FP32 poses to the visible history at prediction.
        prefix = len(sample['condition_viewmats'])
        views = sample['viewmats'][start-prefix:start-prefix+length]
        inputs.temporal['viewmats'][:, start:start+length].copy_(views.to(device=device, dtype=torch.float32)[None])
        return inputs

    def new_interactive_cache(self, model, inputs, frames, condition, extra_sink, *, sink_frames=None):
        return self.new_cache(model, inputs, frames, *condition.shape[-2:],
                              sink_frames=sink_frames)

    def predict_interactive_block(self, model, latent, timestep, context, inputs, start,
                                  cache, *, extra_sink=0, commit=False, target=None):
        return self.predict_block(model, latent, context, inputs, start, timestep,
                                  target if self.uses_anyflow else None, cache=cache, commit=commit)

    @staticmethod
    @torch.amp.autocast('cuda', enabled=False)
    def camera_window(views, intrinsics, start, end, tokens_per_frame):
        # Official order: invert W2C, rebase C2W, then invert back, all in FP32.
        poses = _invert_SE3(views)
        relative = views[:, start:start + 1] @ poses[:, start:end]
        window = _invert_SE3(relative)
        return {
            "viewmats": window.repeat_interleave(tokens_per_frame, dim=1),
            "K": intrinsics[:, start:end].repeat_interleave(tokens_per_frame, dim=1),
        }

    @torch.no_grad()
    def sample_latents(self, model, first_latent, context, views, intrinsics, generator,
                       *, num_latent_frames, sampling_steps, timestep_shift=None,
                       sampling_mode=None, sink_frames=None):
        """Cached flow-map rollout with native camera conditioning."""
        from sampling import denoise_few_step_block
        from wan.conditioning import TemporalInputs

        device = first_latent.device
        chunk = self.num_frame_per_block
        if num_latent_frames <= 0 or num_latent_frames % chunk:
            raise ValueError("SolarWM output length must be a positive multiple of 3")
        channels, height, width = first_latent.shape[2:]
        output = torch.zeros(1, num_latent_frames, channels, height, width,
                             device=device, dtype=torch.bfloat16)
        output[:, 0] = first_latent[:, 0]
        # Stage 1 draws each block immediately before denoising. Stage 2 draws
        # the complete initial trajectory before any transition noise.
        initial_noise = (torch.randn(output.shape, generator=generator, device=device, dtype=output.dtype)
                         if self.initial_noise_full_sequence else None)
        if initial_noise is not None:
            initial_noise[:, 0] = first_latent[:, 0]
        schedule = self.opsd_schedule(device, sampling_steps, timestep_shift=timestep_shift,
                                      sampling_mode=sampling_mode)
        inputs = TemporalInputs({'viewmats': views, 'intrinsics': intrinsics},
                                {'first_latent': first_latent})
        cache = self.new_cache(model, inputs, num_latent_frames, height, width,
                                       sink_frames=sink_frames)
        def restore_first(latent):
            latent[:, 0] = first_latent[:, 0]
            return latent
        for block in trange(num_latent_frames // chunk, desc=f"SolarWM {self.release_name}"):
            start, end = block * chunk, (block + 1) * chunk
            latent = (torch.randn((1, chunk, channels, height, width), generator=generator,
                                  device=device, dtype=output.dtype) if initial_noise is None else
                      initial_noise[:, start:end].clone())
            if block == 0:
                restore_first(latent)
            targets = iter(schedule.targets) if self.uses_anyflow else None
            latent = denoise_few_step_block(
                latent, schedule,
                lambda current, t: self.predict_block(model, current, context, inputs, start, t,
                    None if targets is None else next(targets), cache=cache),
                lambda current: torch.randn(current.shape, generator=generator, device=device, dtype=current.dtype),
                restore=restore_first if block == 0 else None,
            )
            output[:, start:end] = latent
            self.predict_block(model, latent, context, inputs, start, 0.0,
                               0.0 if self.uses_anyflow else None, cache=cache, commit=True)
        return output

    @torch.no_grad()
    def _inference(self, model, samples, text_encoder, codec, seeds, device, **options):
        contexts, negative_contexts = self.inference_contexts(samples, text_encoder, device, **options)
        text_scale = self.inference_text_options(options)['guidance_scale']
        videos, latents, images = [], [], []
        for index, (sample, seed, context) in enumerate(zip(samples, seeds, contexts, strict=True)):
            frames = options['num_latent_frames']
            if frames is None:
                frames = len(sample['condition_latent']) + int(sample['target_frames'])
            first = self._move(sample["condition_latent"], device)[None]
            if tuple(first.shape) != (1, 1, *self.latent_shape):
                raise ValueError(
                    f"SolarWM {self.release_name} requires one 48x30x54 latent image "
                    "(480x864 RGB)"
                )
            camera = []
            for name, shape in (("viewmats", (4, 4)), ("intrinsics", (3, 3))):
                prefix, target = sample[f"condition_{name}"], sample[name]
                if tuple(prefix.shape) != (1, *shape) or len(target) < frames - 1:
                    raise ValueError(f"SolarWM {name} must cover the first image and {frames - 1} targets")
                camera.append(torch.cat((prefix, target[:frames - 1])).to(device=device, dtype=torch.float32)[None])
            negative = None if negative_contexts is None else [negative_contexts[index]]
            with self.unwrap(model).text_guidance(negative, text_scale):
                latent = self.sample_latents(
                    model, first, [context], *camera,
                    torch.Generator(device=device).manual_seed(int(seed)),
                    num_latent_frames=frames, sampling_steps=options["sampling_steps"],
                    timestep_shift=options.get('timestep_shift'),
                    sampling_mode=options.get('sampling_mode'),

                    sink_frames=options.get('sink_frames'),
                )
            video, _ = self.decode_output(codec, latent)
            videos.append(video[0])
            latents.append(latent[0])
            images.append(video[0, 0].clone())
        return torch.stack(videos), torch.stack(latents), torch.stack(images)


class SolarWMFewStep(SolarWM):
    """Released SolarWM-5B-sgf-stage2-81f final NFE4 checkpoint."""

    supports_sft = False
    flow_objective = "flow_matching"
    checkpoint_stage = "stage2"
    checkpoint_objective = "flow_matching"
    release_name = "Stage 2 SGF"
    sampling_steps = 4
    sampling_mode = "sde"
    initial_noise_full_sequence = True

    def model_config(self, args):
        del args
        return {"stage": "stage2"}

    def validate_model_config(self, config):
        if config != {"stage": "stage2"}:
            raise ValueError(
                "SolarWM final few-step model config must be {'stage': 'stage2'}"
            )

    @staticmethod
    def official_weight_key(payload):
        if "generator_ema" not in payload:
            raise ValueError(
                "Released SolarWM Stage 2 inference requires generator_ema"
            )
        return "generator_ema"

    def validate_official_checkpoint(self, payload, config):
        if payload.get("checkpoint_format") not in (
            None, "solarwm.wan22-stage2-sgf.v1",
        ):
            raise ValueError("SolarWM Stage 2 checkpoint format differs from the release")
        model = config.get("model", {})
        train = config.get("train", {})
        expected_model = {
            "family": "wan22_ti2v_5b",
            "causal": True,
            "camera_attention_mode": "fused_prope",
            "camera_translation_transform": "linear",
            "latent_channels": 48,
            "frame_sequence_length": 405,
            "num_frame_per_block": 3,
            "max_prior_clean_chunks": 5,
            "local_attn_size": 18,
            "sink_size": 0,
            "rope_train_frames": 21,
            "use_echorope": False,
        }
        expected_train = {
            "stage": "stage2",
            "causal_mode": "self_gradient_forcing",
            "objective": "flow_matching",
        }
        mismatches = {
            f"model.{name}": (model.get(name), value)
            for name, value in expected_model.items()
            if model.get(name) != value
        }
        mismatches.update({
            f"train.{name}": (train.get(name), value)
            for name, value in expected_train.items()
            if train.get(name) != value
        })
        steps = tuple(train.get("denoising_step_list", ()))
        if steps != DEFAULT_FEW_STEP_TIMESTEPS:
            mismatches["train.denoising_step_list"] = (
                steps, DEFAULT_FEW_STEP_TIMESTEPS,
            )
        if mismatches:
            raise ValueError(
                "SolarWM final few-step checkpoint contract differs "
                f"(actual, expected): {mismatches}"
            )

    def validate_sampling_steps(self, sampling_steps):
        if sampling_steps != 4:
            raise ValueError("SolarWM Stage 2 supports its released NFE4 sampler only")

    def opsd_schedule(self, device, sampling_steps, *, timestep_shift=None, sampling_mode=None):
        self.validate_sampling_steps(sampling_steps)
        shift = self.timestep_shift if timestep_shift is None else float(timestep_shift)
        if shift != 5.0:
            raise ValueError("SolarWM Stage 2 checkpoint requires timestep shift 5.0")
        return make_solar_sgf_scheduler(device, shift=shift,
                                 sampling_mode=self.sampling_mode if sampling_mode is None else sampling_mode)

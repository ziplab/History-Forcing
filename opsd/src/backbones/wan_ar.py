"""Wan autoregressive backend: temporal layout, conditioning and block sampling."""

import math
from abc import abstractmethod
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict

import torch
from torch.distributed.fsdp import FullyShardedDataParallel

from scheduler import DEFAULT_FEW_STEP_TIMESTEPS, parse_sampling_steps, sampling_step
from wan.conditioning import (
    RolloutConditioning,
    TemporalInputs,
    TemporalLayout,
    batch_inputs,
    fixed_camera_inputs,
)
from wan.geometry import camera_trajectory, camera_view_segment, _invert_SE3

from .base import DMDGenerator, CameraControlState, InteractiveSpec, LatentInteractiveSession, STANDARD_PLAYER_ACTIONS, WanBackend
from model import WanModelWrapper


INTERACTIVE_ACTIONS = frozenset(STANDARD_PLAYER_ACTIONS)


def camera_action_trajectory(
    viewmat, action, frames, *, translation_step=0.2, rotation_degrees=10.0,
    yaw_space="local",
):
    """Integrate a standard Player action into a W2C camera trajectory."""
    name = str(action).strip().upper()
    if name not in INTERACTIVE_ACTIONS:
        raise ValueError(
            f"Unsupported interactive action {action!r}; expected "
            + ", ".join(sorted(INTERACTIVE_ACTIONS))
        )
    if type(frames) is not int or frames <= 0:
        raise ValueError("Camera action trajectory needs a positive frame count")
    if not math.isfinite(translation_step) or translation_step <= 0:
        raise ValueError("Interactive translation step must be finite and positive")
    if not math.isfinite(rotation_degrees) or rotation_degrees <= 0:
        raise ValueError("Interactive rotation step must be finite and positive")
    if yaw_space not in {'local', 'world'}:
        raise ValueError('Camera yaw space must be local or world')
    pose = _invert_SE3(viewmat.float())
    translation = {
        "W": (0.0, 0.0, 1.0), "S": (0.0, 0.0, -1.0),
        "A": (-1.0, 0.0, 0.0), "D": (1.0, 0.0, 0.0),
    }.get(name)
    rotation = {
        "LEFT": (1, -1.0), "RIGHT": (1, 1.0),
        "UP": (0, -1.0), "DOWN": (0, 1.0),
    }.get(name)
    delta = None if translation is None else pose.new_tensor(translation).mul_(translation_step / frames)
    update = None
    if rotation is not None:
        axis, sign = rotation
        angle = math.radians(rotation_degrees) * sign / frames
        c, s = math.cos(angle), math.sin(angle)
        update = torch.eye(3, dtype=pose.dtype, device=pose.device)
        if axis == 1:
            update[0, 0], update[0, 2] = c, s
            update[2, 0], update[2, 2] = -s, c
        else:
            update[1, 1], update[1, 2] = c, -s
            update[2, 1], update[2, 2] = s, c
    result = []
    for _ in range(frames):
        if delta is not None:
            pose[:3, 3].add_(pose[:3, :3] @ delta)
        if update is not None:
            pose[:3, :3] = (update @ pose[:3, :3] if axis == 1 and yaw_space == 'world'
                            else pose[:3, :3] @ update)
        result.append(_invert_SE3(pose))
    return torch.stack(result)


def player_camera_block(backend, viewmat, action, count, target_start, *,
                        translation_step=None, rotation_degrees=None):
    """Apply one Player block with the same native frame strides as online inference."""
    if type(count) is not int or count <= 0 or type(target_start) is not int or target_start < 0:
        raise ValueError('Player camera block needs positive count and nonnegative target_start')
    translation = float(backend.player_translation_step if translation_step is None else translation_step)
    rotation = float(backend.player_rotation_degrees if rotation_degrees is None else rotation_degrees)
    name = str(action).strip().upper()
    if name not in INTERACTIVE_ACTIONS:
        raise ValueError(f'Unsupported interactive action {action!r}')
    native_action = ({'UP': 'DOWN', 'DOWN': 'UP'}.get(name, name)
                     if backend.player_pitch_sign == -1 else name)
    strides = [backend.player_first_frame_stride if target_start + i == 0
               else backend.player_frame_stride for i in range(count)]
    denominator = count if backend.player_action_full_block else backend.num_frame_per_block
    if backend.player_action_full_block:
        views = camera_action_trajectory(
            viewmat, native_action, count, translation_step=translation,
            rotation_degrees=rotation, yaw_space=backend.player_yaw_space,
        )
    else:
        views = []
        for stride in strides:
            viewmat = camera_action_trajectory(
                viewmat, native_action, 1,
                translation_step=translation * stride / denominator,
                rotation_degrees=rotation * stride / denominator,
                yaw_space=backend.player_yaw_space,
            )[0]
            views.append(viewmat)
        views = torch.stack(views)
    yaw_delta = (sum(strides) * rotation / denominator *
                 (-1 if name == 'LEFT' else 1) if name in {'LEFT', 'RIGHT'} else 0.0)
    return views, yaw_delta


class WanCausalModelWrapper(WanModelWrapper):
    """Adapt the shared [B,F,C,H,W] flow contract to a native causal DiT."""

    def forward(self, latent, timestep, context, *, clean_latent=None, image_condition=None, **conditions):
        frames, height, width = latent.shape[1], *latent.shape[-2:]
        if clean_latent is not None:
            conditions['clean_x'] = clean_latent.transpose(1, 2)
        flow = super().forward(
            latent.transpose(1, 2), t=timestep, context=context,
            seq_len=frames * height * width // math.prod(self.model.patch_size),
            image_condition=None if image_condition is None else image_condition.transpose(1, 2),
            **conditions,
        )
        return flow.transpose(1, 2)

    def new_cache(self, batch, frames, height, width, device, dtype, extra_sink_size=0,
                  capacity_frames=None, sink_frames=None):
        from wan.kv_cache.causal import make_causal_cache
        return self.guided_cache(lambda: make_causal_cache(
            self.model, batch, frames, height, width, device, dtype,
            extra_sink_size=extra_sink_size, capacity_frames=capacity_frames,
             sink_frames=sink_frames))


class Action2VDMDGenerator(DMDGenerator):
    """DMD capability for backends using the shared Action2V cache wrapper."""

    def dmd_rollout(self, model, rows, context, scheduler, exit_index, device,
                    *, gradient_frames=None, sink_frames=None):
        return _dmd_rollout(
            self, model, rows, context, scheduler, exit_index, device,
            gradient_frames=gradient_frames, sink_frames=sink_frames,
        )

    def dmd_schedule(self, device, timesteps, *, sampling_mode="sde"):
        from scheduler import make_sampling_scheduler
        return make_sampling_scheduler(
            device, timesteps, solver="flow_match", timestep_shift=self.timestep_shift,
            sampling_mode=sampling_mode,
        )

    def dmd_layout(self, condition_frames, target_frames):
        return (
            condition_frames,
            self.temporal_layout.complete_target_frames(
                target_frames, prefix_frames=condition_frames,
            ),
        )


class WanAR(WanBackend):
    """Shared Wan autoregressive training, conditioning and block sampling."""

    @property
    @abstractmethod
    def base(self):
        """Native Wan architecture, explicitly selected by each concrete backend."""

    num_frame_per_block = None
    local_attn_size = None
    sink_size = None
    camera_control = None
    default_latent_frames = None
    independent_first_frame = False
    image_condition_sink = True
    prefill_condition = True
    supports_opsd = True
    sampling_steps = DEFAULT_FEW_STEP_TIMESTEPS
    sampling_mode = "sde"
    guidance_scale = 1.0
    sampling_steps_type = staticmethod(parse_sampling_steps)
    inference_frames_include_condition = False
    camera_reference = "initial"
    color_correction_strength = 0.0
    player_translation_step = 0.2
    player_rotation_degrees = 10.0
    player_frame_stride = 1
    player_first_frame_stride = 1
    player_pitch_sign = 1
    player_yaw_space = 'local'
    player_action_full_block = False
    interactive_clamp_first_frame = True
    initial_noise_full_sequence = False
    interactive_noise_channel_first = False
    interactive_variable_size = False
    sampling_steps_option = 'sampling_steps'
    timestep_shift_option = 'timestep_shift'

    @property
    def temporal_layout(self):
        return TemporalLayout(
            self.num_frame_per_block, self.independent_first_frame, self.image_condition_sink,
            self.prefill_condition,
        )

    def validate_sampling_steps(self, sampling_steps):
        if (type(sampling_steps) is int) != (type(self.sampling_steps) is int):
            raise ValueError(f"{type(self).__name__} requires the same sampling_steps form as its native default")

    def opsd_schedule(self, device, sampling_steps, *, timestep_shift=None, sampling_mode=None):
        """Supply the native sampler; the recipe owns neither its table nor update."""
        from scheduler import make_sampling_scheduler
        self.validate_sampling_steps(sampling_steps)
        return make_sampling_scheduler(
            device, sampling_steps,
            timestep_shift=self.timestep_shift if timestep_shift is None else timestep_shift,
            sampling_mode=self.sampling_mode if sampling_mode is None else sampling_mode,
        )

    def inference_schedule(self, device, options):
        """Use the checkpoint recipe's solver, preserving each native grid factory."""
        if self.inference_recipe(options) == "dmd":
            backend = self
            shift = options.get("timestep_shift")
            shift = self.timestep_shift if shift is None else shift
            if shift != self.timestep_shift:
                backend = copy(self)
                backend.timestep_shift = shift
            steps = options.get("sampling_steps")
            if steps is None:
                raise ValueError("DMD inference needs resolved or explicit sampling_steps")
            mode = options.get("sampling_mode")
            return backend.dmd_schedule(device, steps,
                sampling_mode="sde" if mode is None else mode)
        steps = options.get(self.sampling_steps_option)
        return self.opsd_schedule(
            device, self.sampling_steps if steps is None else steps,
            timestep_shift=options.get(self.timestep_shift_option),
            sampling_mode=options.get("sampling_mode"),
        )

    def inference_recipe(self, options):
        recipe = options.get('sampling_recipe')
        recipe = getattr(self, 'sampling_recipe', 'native') if recipe is None else recipe
        if recipe not in ('native', 'dmd'):
            raise ValueError('sampling_recipe must be native or dmd')
        if recipe == 'dmd' and not isinstance(self, DMDGenerator):
            raise ValueError(f'{type(self).__name__} does not support the DMD generator sampler')
        return recipe

    def inference_text_options(self, options):
        """Resolve native text defaults for offline and interactive inference."""
        scale = options.get('guidance_scale')
        default_scale = 1.0 if self.inference_recipe(options) == 'dmd' else self.guidance_scale
        scale = default_scale if scale is None else scale
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('guidance_scale must be finite and non-negative')
        negative = options.get('negative_prompt')
        negative = self.negative_prompt if negative is None else negative
        if not isinstance(negative, str):
            raise TypeError('negative_prompt must be a string')
        return {'guidance_scale': float(scale), 'negative_prompt': negative}

    def inference_contexts(self, samples, text_encoder, device, **options):
        text = self.inference_text_options(options)
        positive, _ = super().inference_contexts(samples, text_encoder, device, **options)
        if text['guidance_scale'] == 1.0:
            return positive, None
        return positive, text_encoder([text['negative_prompt']] * len(samples), device)


    def flow_forward(self, model, batch, context, device, schedule, **options):
        batch = self.trim_training_targets(batch)
        clean = self._move(batch["clean_latent"], device)
        condition, condition_frames = self.training_condition(batch, device)
        from scheduler import sample_flow_training

        generated_prefix = self.temporal_layout.generated_prefix_frames(condition_frames)
        if generated_prefix:
            from dataset import unbatch_latent_sequences
            clean = torch.cat((condition, clean), dim=1)
            inputs = batch_inputs([
                self.prepare_inputs(row, len(row['clean_latent']), device).pad(clean.shape[1])
                for row in unbatch_latent_sequences(batch)
            ])
            prefix_inputs = None
        else:
            prefix_inputs, inputs = self.training_inputs(batch, device)

        noisy, target, timestep, weight = sample_flow_training(
            clean, self.temporal_layout.target_blocks(clean.shape[1] - generated_prefix,
                                                    prefix_frames=condition_frames), schedule
        )
        if generated_prefix:
            noisy[:, :generated_prefix] = condition
            timestep[:, :generated_prefix] = 0
        secondary_context = options.get("secondary_context")
        prompt_mask = batch.get("teacher_prompt_mask")
        if (secondary_context is None) != (prompt_mask is None):
            raise ValueError(
                "Teacher context and prompt mask must be provided together"
            )
        if prompt_mask is not None:
            if generated_prefix:
                prompt_mask = torch.cat((prompt_mask[:, :generated_prefix], prompt_mask), dim=1)
            prompt_mask = prompt_mask.to(
                device=device, non_blocking=True
            )
        prediction = model(
            noisy,
            timestep,
            context,
            clean_latent=clean,
            condition_latent=condition if self.prefill_condition else None,
            condition_inputs=prefix_inputs.slice(0, condition_frames) if condition_frames and self.prefill_condition else None,
            secondary_context=secondary_context,
            prompt_mask=prompt_mask,
            **inputs.forward_kwargs(),
        )
        valid_mask = batch["valid_frame_mask"].to(
            device=device, non_blocking=True
        )
        if generated_prefix:
            prediction, target, weight, clean = (
                value[:, generated_prefix:] for value in (prediction, target, weight, clean)
            )
        return self.masked_flow_output(
            prediction, target, weight, clean, valid_mask
        )

    def interactive_session(
        self, model, text_encoder, codec, device, **options,
    ):
        """Create the shared cached session used by Wan AR backbones."""
        return WanARInteractiveSession(
            self, model, text_encoder, codec, device, **options,
        )

    def interactive_spec(self):
        if self.camera_control:
            mode = "camera"
        elif "actions" in self.required_data_fields:
            mode = "discrete"
        else:
            mode = "none"
        return InteractiveSpec(
            actions=STANDARD_PLAYER_ACTIONS,
            block_frames=self.num_frame_per_block,
            control_mode=mode,
            condition_consumes_block_slot=not self.prefill_condition,
            control_features=(
                getattr(self, "interactive_control_features", None)
                if mode == "discrete" else None
            ),
        )

    def interactive_trajectory(self, actions, *, condition_frames=1):
        """Prepare native block controls using the online Player's camera steps."""
        spec = self.interactive_spec()
        if spec.control_mode == "none":
            return super().interactive_trajectory(actions, condition_frames=condition_frames)
        if isinstance(actions, str):
            raise ValueError("Native actions must be a sequence of block names")
        actions = tuple(actions)
        if not actions or any(not isinstance(action, str) or action not in spec.actions
                              for action in actions):
            raise ValueError("Native actions must contain supported block names")
        prefix = self.temporal_layout.generated_prefix_frames(condition_frames)
        frames = len(actions) * spec.block_frames + condition_frames - prefix
        view = torch.eye(4, dtype=torch.float32, device="cpu")
        views = [view] * condition_frames
        states = torch.zeros(frames, spec.control_features, dtype=torch.int8, device="cpu") if (
            spec.control_mode == "discrete") else None
        yaw = 0.0
        for index, action in enumerate(actions):
            start = index * spec.block_frames + condition_frames - prefix
            first = max(start, condition_frames)
            count = start + spec.block_frames - first
            if not count:
                continue
            if states is not None:
                state = self.interactive_action_state(action, device="cpu", dtype=torch.float32)
                states[first:first + count] = state.to(dtype=states.dtype)

            generated, delta = player_camera_block(
                self, view, action, count, first - condition_frames)
            views.extend(generated)
            view = generated[-1]
            yaw += delta
        intrinsics = torch.as_tensor(self.interactive_input_options().get(
            "intrinsics", ((1., 0., .5), (0., 1., .5), (0., 0., 1.))),
            dtype=torch.float32, device="cpu")
        if intrinsics.shape != (3, 3):
            raise ValueError("Interactive intrinsics must be a normalized 3x3 matrix")
        return {**({"actions": states.numpy()} if states is not None else {}),
                "viewmats": torch.stack(views).numpy(),
                "intrinsics": intrinsics[None].repeat(frames, 1, 1).numpy(),
                "rgb_frame_indices": torch.tensor([
                    self.decoded_frames(index + 1) - 1 for index in range(frames)
                ], dtype=torch.int64, device="cpu").numpy(),
                "commanded_yaw_degrees": yaw}

    def prepare_interactive_sample(self, sample, codec, device, **options):
        """Restore any native visual inputs before starting an online episode."""
        return sample

    def prepare_interactive_inputs(self, sample, target_frames, device, dtype):
        return self.prepare_inputs(sample, target_frames, device, dtype=dtype)

    def update_interactive_camera(self, inputs, sample, start, length, device, dtype):
        """Update a camera segment while preserving mechanism-specific preparation."""
        if getattr(self.prepare_inputs, '__func__', None) is not WanAR.prepare_inputs:
            return self.prepare_interactive_inputs(sample, len(sample['viewmats']), device, dtype)
        views = camera_view_segment(
            sample, start, length, device, dtype,
            reference=self.camera_reference, block_frames=self.num_frame_per_block)
        inputs.temporal['viewmats'][:, start:start+length].copy_(views)
        return inputs

    def new_interactive_cache(self, model, inputs, frames, condition, extra_sink, *, sink_frames=None):
        return self.unwrap(model).new_cache(
            1, frames, *condition.shape[-2:], condition.device, condition.dtype, extra_sink,
             sink_frames=sink_frames,
        )

    def predict_interactive_block(self, model, latent, timestep, context, inputs, start,
                                  cache, *, extra_sink=0, commit=False, target=None):
        """Use the normal flow wrapper; the session owns only time and cache state."""
        inner = self.unwrap(model).model
        position = _position(start, latent, inner)
        return model(
            latent, timestep, context, **inputs.slice(start, latent.shape[1]).forward_kwargs(),
            **cache.forward_kwargs(), current_start=position, position_start=position,
            extra_sink_size=extra_sink, commit_cache=commit,
        )

    def interactive_action_state(self, action, *, device, dtype):
        raise NotImplementedError('Discrete backends must declare their native action mapping')

    def training_inputs(self, batch, device):
        """Prepare named prefix/target inputs; pad only complete masked AR tails."""
        from dataset import unbatch_latent_sequences
        rows = unbatch_latent_sequences(batch)
        frames = batch['clean_latent'].shape[1]
        prefix_frames = len(rows[0]['condition_latent'])
        prepared = []
        for row in rows:
            inputs = self.prepare_inputs(row, len(row['clean_latent']), device)
            prepared.append(inputs.pad(prefix_frames + frames))
        inputs = batch_inputs(prepared)
        return inputs.slice(0, prefix_frames), inputs.slice(prefix_frames, frames)

    def sample_latents(
        self,
        model,
        noise,
        context,
        negative_context,
        inputs,
        prefix,
        generators,
        options,
    ):
        """Run the shared AR sampler with optional native text CFG."""
        scheduler = self.inference_schedule(noise.device, options)
        return model.sample_causal_sequence(
            noise,
            context,
            inputs,
            scheduler,
            prefix=prefix,
            negative_context=negative_context,
            guidance_scale=self.inference_text_options(options)['guidance_scale'],
            noise_generators=generators,
            noise_scale=options["noise_scale"],

            sink_frames=options.get("sink_frames"),

        )

    @torch.no_grad()
    def _generate(
        self, model, samples, text_encoder, codec, seeds, device, **options
    ):
        dtype = options.get("dtype", torch.bfloat16)
        context, negative_context = self.inference_contexts(
            samples, text_encoder, device, **options
        )
        conditions, condition_frames = self.inference_conditions(samples)
        requested_frames = options["num_latent_frames"]
        frames = int(samples[0]['target_frames'] if requested_frames is None else requested_frames)
        if requested_frames is None and any(int(row['target_frames']) != frames for row in samples):
            raise ValueError('Data-length inference batches must share a target length')
        if self.inference_frames_include_condition and requested_frames is not None:
            frames -= condition_frames
        self.temporal_layout.target_blocks(frames, prefix_frames=condition_frames)
        inputs = batch_inputs([
            self.prepare_inputs(sample, frames, device, dtype=dtype) for sample in samples
        ])
        generated_prefix = self.temporal_layout.generated_prefix_frames(condition_frames)
        target_inputs = inputs.slice(condition_frames - generated_prefix, frames + generated_prefix)
        noise, generators = self.initial_noise(
            seeds,
            (frames + generated_prefix, *self.latent_shape),
            device,
            options["noise_scale"],
        )
        prefix = condition = None
        if condition_frames:
            prefix_frames = condition_frames
            condition = torch.stack([
                value[:prefix_frames] for value in conditions
            ]).to(device=device, dtype=dtype)
            prefix = model.prepare_condition(condition, inputs.slice(0, prefix_frames))
        latent = self.sample_latents(
            model,
            noise,
            context,
            negative_context,
            target_inputs,
            prefix,
            generators,
            options,
        )
        return latent, condition

    @torch.no_grad()
    def _inference(
        self, model, samples, text_encoder, codec, seeds, device, **options
    ):
        latents, condition = self._generate(
            model, samples, text_encoder, codec, seeds, device, **options
        )
        if self.output_includes_condition:
            if self.prefill_condition and condition is not None:
                latents = torch.cat((condition, latents), 1)
            videos = (self.decode(codec, latents) * .5 + .5).clamp(0, 1)
            condition_images = videos[:, 0].clone() if condition is not None else None
        else:
            videos, condition_images = self.decode_output(codec, latents, condition)
        strength = options.get("color_correction_strength", self.color_correction_strength)
        if strength:
            from utils import chunk_color_correct
            videos = chunk_color_correct(videos, strength, self.num_frame_per_block,
                                          condition_frames=len(condition[0]))
        return videos, latents, condition_images

    def decoded_output_frames(self, latent_frames, condition_frames=0):
        if self.output_includes_condition:
            return self.decoded_frames(latent_frames)
        return super().decoded_output_frames(latent_frames, condition_frames)

    def build(self, model_dir, config):
        self.validate_model_config(config)
        from model import Action2VModel

        architecture = self.architecture_config()
        architecture.pop("num_frame_per_block")
        return Action2VModel(
            model_dir, **architecture, temporal_layout=self.temporal_layout,
            transformer_type=self.transformer_class(), base=self.base,
        )

    @staticmethod
    def transformer_class():
        from wan.causal import CausalWanModel

        return CausalWanModel

    def condition_block_lengths(self, frames):
        return self.temporal_layout.prefix_blocks(frames)

    def validate_training_lengths(self, condition_lengths, target_lengths):
        prefixes = set(condition_lengths)
        if len(prefixes) != 1:
            raise ValueError("AR training requires one condition length within a batch")
        generated_prefix = self.temporal_layout.generated_prefix_frames(next(iter(prefixes)))
        if min(target_lengths) + generated_prefix < self.num_frame_per_block:
            raise ValueError("Targets must contain at least one complete AR block")

    def training_batch_key(self, condition_frames, target_frames):
        return condition_frames

    def prepare_inputs(self, sample, target_frames, device, *, dtype=torch.bfloat16):
        """Prepare native camera conditions before batching and block slicing."""
        if not self.camera_control:
            return TemporalInputs()
        views, intrinsics = camera_trajectory(
            sample, target_frames, device, dtype,
            reference=self.camera_reference, block_frames=self.num_frame_per_block)
        static = ({"initial_frame": self._move(sample["condition_latent"][None], device)}
                  if not self.prefill_condition else {})
        return TemporalInputs({"viewmats": views, "intrinsics": intrinsics}, static)

    def fixed_view_inputs(self, inputs, prefix_frames, target_frames):
        return fixed_camera_inputs(inputs, prefix_frames, target_frames,
                                   reference=self.camera_reference)

    def opsd_conditioning(self, samples, target_frames, device):
        """Batch native inputs once, then derive the fixed teacher and block views.

        Samples share a rollout layout. Unchanged teacher inputs (such as image
        conditions and reference slots) share storage with the student batch.
        """
        prefix = len(samples[0]["condition_latent"])
        generated_prefix = self.temporal_layout.generated_prefix_frames(prefix)
        inputs = batch_inputs([self.prepare_inputs(sample, target_frames, device) for sample in samples])
        teacher = self.fixed_view_inputs(inputs, prefix, target_frames)
        target_start = prefix - generated_prefix
        target_length = target_frames + generated_prefix
        return RolloutConditioning(
            inputs.slice(0, target_start),
            inputs.slice(target_start, target_length),
            teacher.slice(target_start, target_length),
        )


    @staticmethod
    def add_cache_arguments(parser):
        from wan.kv_cache.causal import resolve_sink_frames

        parser.add_argument('--sink-frames', type=resolve_sink_frames, default=None,
                            help='Override the native temporal sink frame count')

    def add_common_inference_arguments(self, parser):

        self.add_cache_arguments(parser)
        self.add_length_arguments(parser)
        parser.add_argument("--sampling-mode", choices=("ode", "sde"), default=None)
        if isinstance(self, DMDGenerator):
            parser.add_argument("--sampling-recipe", choices=("native", "dmd"), default=None,
                                help="Choose solver/grid/CFG defaults; mode and shift resolve separately")
        parser.add_argument("--timestep-shift", type=float, default=None,
                            help="Override the checkpoint or native timestep shift")
        parser.add_argument(
            "--noise-scale", type=float, default=1.0,
            help="Standard deviation multiplier for sampler Gaussian noise",
        )

    def add_length_arguments(self, parser):
        parser.add_argument(
            "--num-latent-frames", type=int, default=self.default_latent_frames
        )
        parser.add_argument(
            "--use-data-length", action="store_true",
            help="Generate each packed sample's target_frames instead of one global length",
        )

    def inference_length_options(self, args):
        if args.use_data_length or args.num_latent_frames is None:
            return {"num_latent_frames": None}
        if self.inference_frames_include_condition:
            self.temporal_layout.target_blocks(args.num_latent_frames - 1, prefix_frames=1)
        else:
            self.target_block_lengths(args.num_latent_frames)
        return {"num_latent_frames": args.num_latent_frames}

    def add_inference_arguments(self, parser):
        self.add_common_inference_arguments(parser)
        if self.color_correction_strength:
            parser.add_argument("--color-correction-strength", type=float, default=self.color_correction_strength)
        parser.add_argument(
            "--sampling-steps",
            type=parse_sampling_steps,
            default=None,
            help="Denoising step count or explicit decreasing timesteps; defaults to checkpoint/native settings",
        )

    def common_inference_options(self, args):
        if not math.isfinite(args.noise_scale) or args.noise_scale <= 0:
            raise ValueError("noise_scale must be finite and positive")
        recipe = self.inference_recipe({'sampling_recipe': getattr(args, 'sampling_recipe', None)})
        shift = getattr(args, "timestep_shift", None)
        shift = self.timestep_shift if shift is None else shift
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("timestep_shift must be finite and positive")
        return {
            "noise_scale": float(args.noise_scale),
            "sampling_mode": getattr(args, "sampling_mode", None) or ("sde" if recipe == "dmd" else self.sampling_mode),
            **self.inference_length_options(args),

            "sink_frames": getattr(args, 'sink_frames', None),
            "sampling_recipe": recipe,
            "timestep_shift": float(shift),
        }


    def inference_options(self, args):
        steps = self.sampling_steps if args.sampling_steps is None else args.sampling_steps
        options = self.common_inference_options(args)
        if options["sampling_recipe"] == "dmd":
            if args.sampling_steps is None:
                raise ValueError("DMD inference needs resolved or explicit sampling_steps")
            steps = parse_sampling_steps(steps)
        else:
            self.validate_sampling_steps(steps)
        strength = getattr(args, "color_correction_strength", self.color_correction_strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Color correction strength must be in [0,1]")
        return {
            **options,
            "sampling_steps": steps,
            "color_correction_strength": strength,
        }

    def architecture_config(self):
        values = {
            "num_frame_per_block": self.num_frame_per_block,
            "local_attn_size": self.local_attn_size,
            "sink_size": self.sink_size,
        }
        if (
            type(values["num_frame_per_block"]) is not int
            or values["num_frame_per_block"] <= 0
            or type(values["local_attn_size"]) is not int
            or values["local_attn_size"] == 0
            or values["local_attn_size"] < -1
            or type(values["sink_size"]) is not int
            or values["sink_size"] < 0
        ):
            raise TypeError("Concrete Wan backbones must define architecture constants")
        local_attention = values["local_attn_size"]
        sink_size = values["sink_size"]
        if local_attention == -1 and sink_size:
            raise ValueError("Global attention cannot define sink frames")
        if local_attention != -1 and sink_size >= local_attention:
            raise ValueError("sink_size must be smaller than local_attn_size")
        return values

    def model_config(self, args):
        self.checkpoint_file(args.checkpoint)
        self.architecture_config()
        return {}

    def training_identity(self):
        return {
            **super().training_identity(),
            "architecture": self.architecture_config(),
            "temporal_layout": asdict(self.temporal_layout),
        }




    def validate_model_config(self, config):
        if not isinstance(config, dict) or config:
            raise ValueError("Model config must be an empty mapping")
        self.architecture_config()
        return config


    def trim_training_targets(self, batch):
        """Expose only complete native AR blocks without changing packed data."""
        mask = batch["valid_frame_mask"]
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("valid_frame_mask must be a boolean [B,F] tensor")
        lengths = mask.sum(dim=1)
        layout = self.temporal_layout
        prefix_frames = batch["condition_latent"].shape[1]
        usable = lengths.new_tensor([
            layout.complete_target_frames(length, prefix_frames=prefix_frames)
            for length in lengths.tolist()
        ])
        if torch.any(usable <= 0):
            raise ValueError("A training row has no complete native AR block")
        frames = int(usable.max().item())
        trimmed = dict(batch)
        trimmed["clean_latent"] = batch["clean_latent"][:, :frames]
        trimmed["valid_frame_mask"] = (
            torch.arange(frames, device=mask.device)[None] < usable[:, None]
        )
        for name in ("viewmats", "intrinsics"):
            if name in batch:
                trimmed[name] = batch[name][:, :frames]
        if "teacher_prompt_mask" in batch:
            trimmed["teacher_prompt_mask"] = batch[
                "teacher_prompt_mask"
            ][:, :frames]
        return trimmed


    def schedule(self, device, config=None):
        del config
        from scheduler import make_flow_training_schedule

        return make_flow_training_schedule(device, self.timestep_shift)


    def condition_frame_count(self, conditions):
        """Validate one image latent or complete video-prefix blocks."""
        lengths = [len(condition) for condition in conditions]
        if not lengths or len(set(lengths)) != 1 or lengths[0] <= 0:
            raise ValueError(
                "Every row in a conditioned batch must have the same "
                "positive latent length"
            )
        frames = lengths[0]
        self.condition_block_lengths(frames)
        return frames

    def inference_conditions(self, samples):
        """Use the shared packed condition without changing the target timeline."""
        conditions = [sample["condition_latent"] for sample in samples]
        present = [
            value is not None and value.shape[0] > 0 for value in conditions
        ]
        if any(present) and not all(present):
            raise ValueError("An inference batch must share condition type")
        condition_frames = (
            self.condition_frame_count(conditions) if present[0] else 0
        )
        if condition_frames and self.camera_control:
            invalid = [
                index
                for index, sample in enumerate(samples)
                if (
                    sample["condition_viewmats"].shape[0] != condition_frames
                    or sample["condition_intrinsics"].shape[0]
                    != condition_frames
                )
            ]
            if invalid:
                raise ValueError(
                    "Condition latent and camera metadata lengths differ for "
                    f"rows {invalid}"
                )
        return conditions, condition_frames

    def target_block_lengths(self, frames, *, condition_frames=1):
        """Return this backend's target partition for one AR rollout."""
        return self.temporal_layout.target_blocks(frames, prefix_frames=condition_frames)

    def training_condition(self, batch, device):
        condition = batch.get("condition_latent")
        if condition is None or not condition.shape[1]:
            return None, 0
        condition = self._move(condition, device)
        lengths = batch["condition_mask"].sum(dim=1)
        if not torch.all(lengths > 0):
            raise ValueError(
                "A training batch cannot mix conditioned and unconditioned rows"
            )
        frames = self.condition_frame_count([
            condition[row, :int(length)]
            for row, length in enumerate(lengths.tolist())
        ])
        return condition[:, :frames], frames


class WanARInteractiveSession(LatentInteractiveSession):
    """One cached episode runner for Wan AR, discrete control and flow-map models."""

    def __init__(self, backend, model, text_encoder, codec, device, *, sink_frames=None,
                 dtype=torch.bfloat16, player_translation_step=None,
                 player_rotation_degrees=None, **options):
        super().__init__(backend, codec)
        self.sink_frames = sink_frames
        self.model, self.text_encoder = model, text_encoder
        self.device, self.dtype = torch.device(device), dtype
        noise_scale = options.get('noise_scale')
        noise_scale = 1.0 if noise_scale is None else noise_scale
        if not math.isfinite(noise_scale) or noise_scale <= 0:
            raise ValueError('noise_scale must be finite and positive')
        self.options = {**options, **backend.inference_text_options(options), 'noise_scale': float(noise_scale)}
        self.spec = backend.interactive_spec()
        self.translation_step = float(backend.player_translation_step if player_translation_step is None else player_translation_step)
        self.rotation_degrees = float(backend.player_rotation_degrees if player_rotation_degrees is None else player_rotation_degrees)
        if (not math.isfinite(self.translation_step) or self.translation_step <= 0
                or not math.isfinite(self.rotation_degrees) or self.rotation_degrees <= 0):
            raise ValueError('Interactive camera steps must be finite and positive')
        self._clear_episode()

    def _clear_episode(self):
        self._cache = self._negative_cache = None
        self._inputs = self._condition = self._context = self._negative_context = self._sample = None
        self._generator = self._initial_noise = self._schedule = self._last_view = None
        self._steps = self._max_steps = 0
        self._commanded_yaw = 0.0
        super()._clear_episode()

    @torch.no_grad()
    def reset(self, sample, *, seed, max_steps):
        self._clear_episode()
        with self._episode_update():
            return self._reset(sample, seed=seed, max_steps=max_steps)

    def _reset(self, sample, *, seed, max_steps):
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError('Interactive max_steps must be a positive integer')
        condition = sample.get('condition_latent')
        if condition is None or condition.ndim != 4 or not len(condition):
            raise ValueError('Interactive inference requires a visual condition')
        self._prefix_frames = len(condition)
        self._generated_prefix = self.backend.temporal_layout.generated_prefix_frames(self._prefix_frames)
        self._extra_sink = self.backend.temporal_layout.extra_sink_frames(self._prefix_frames)
        total = max_steps * self.spec.block_frames + self._prefix_frames - self._generated_prefix
        target_frames = total - self._prefix_frames
        row = dict(sample)
        if self.spec.control_mode == 'camera':
            views, ks = row.get('condition_viewmats'), row.get('condition_intrinsics')
            if self.spec.control_mode == 'camera' and (
                views is None or views.shape != (1, 4, 4) or ks is None or ks.shape != (1, 3, 3)
            ):
                raise ValueError('Interactive camera control needs one calibrated condition view')
            if views is None:
                views = torch.eye(4).repeat(self._prefix_frames, 1, 1)
            if ks is None:
                ks = torch.eye(3).repeat(self._prefix_frames, 1, 1)
            row['condition_viewmats'], row['condition_intrinsics'] = views, ks
            self._last_view = views[-1].float().cpu().clone()
            row['viewmats'] = self._last_view[None].expand(target_frames, -1, -1).clone()
            row['intrinsics'] = ks[-1:].float().cpu().expand(target_frames, -1, -1).clone()
        if self.spec.control_mode == 'discrete':
            row['actions'] = torch.zeros(target_frames, self.spec.control_features)
        row = self.backend.prepare_interactive_sample(row, self.codec, self.device, **self.options)
        self._sample = row
        self._condition = row['condition_latent'][None].to(device=self.device, dtype=self.dtype)
        shape = tuple(self._condition.shape[2:])
        expected = self.backend.latent_shape
        if shape[0] != expected[0] or (not self.backend.interactive_variable_size and shape != expected):
            raise ValueError(f'Interactive condition shape {shape} differs from {expected}')
        self._inputs = self.backend.prepare_interactive_inputs(row, target_frames, self.device, self.dtype)
        self._context, self._negative_context = self.backend.inference_contexts(
            [row], self.text_encoder, self.device, **self.options,
        )
        self._cache = self.backend.new_interactive_cache(
            self.model, self._inputs, total, self._condition, self._extra_sink,
             sink_frames=self.sink_frames,
        )
        self._negative_cache = (None if self._negative_context is None else
            self.backend.new_interactive_cache(self.model, self._inputs, total,
                                               self._condition, self._extra_sink,
                                               sink_frames=self.sink_frames))
        start = 0
        for length in self.backend.temporal_layout.prefix_blocks(self._prefix_frames):
            self._commit(self._condition[:, start:start + length], start)
            start += length
        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self._initial_noise = None
        if self.backend.initial_noise_full_sequence:
            self._initial_noise = self._draw_noise((1, max_steps * self.spec.block_frames, *shape))
        self._steps, self._max_steps, self._commanded_yaw = 0, max_steps, 0.0
        self._schedule = self.backend.inference_schedule(self.device, self.options)
        self._start_rendering(self._condition)
        return self.observe()

    def _draw_noise(self, shape):
        scale = self.options['noise_scale']
        if self.backend.interactive_noise_channel_first:
            batch, frames, channels, height, width = shape
            noise = torch.randn((batch, channels, frames, height, width),
                                generator=self._generator, device=self.device, dtype=self.dtype)
            return noise.transpose(1, 2).mul_(scale)
        return torch.randn(shape, generator=self._generator, device=self.device, dtype=self.dtype).mul_(scale)

    def _flow(self, latent, times, start, context, cache, *, commit=False, target=None):
        return self.backend.predict_interactive_block(
            self.model, latent, times, context, self._inputs, start, cache,
            extra_sink=self._extra_sink, commit=commit, target=target,
        )

    def _commit(self, latent, start):
        inner = self.backend.unwrap(self.model).model
        times = torch.full((1, latent.shape[1]), getattr(inner, 'context_timestep', 0.),
                           device=self.device, dtype=torch.float32)
        self._flow(latent, times, start, self._context, self._cache, commit=True, target=0.)
        self._cache.validate_detached()
        if self._negative_cache is not None:
            self._flow(latent, times, start, self._negative_context, self._negative_cache,
                       commit=True, target=0.)
            self._negative_cache.validate_detached()

    def _set_action(self, name, start, length):
        first = max(start, self._prefix_frames)
        count = start + length - first
        target_start = first - self._prefix_frames
        if self.spec.control_mode == 'discrete':
            state = self.backend.interactive_action_state(name, device=self.device, dtype=self.dtype)
            self._inputs.temporal['actions'][:, first:first + count].copy_(state[None, None])
        elif self.spec.control_mode == 'camera' and count:
            views, yaw_delta = player_camera_block(
                self.backend, self._last_view, name, count, target_start,
                translation_step=self.translation_step,
                rotation_degrees=self.rotation_degrees,
            )
            self._last_view = views[-1]
            self._sample['viewmats'][target_start:target_start + count] = views
            self._inputs = self.backend.update_interactive_camera(
                self._inputs, self._sample, first, count, self.device, self.dtype,
            )
            self._commanded_yaw += yaw_delta

    @torch.no_grad()
    def step(self, action, *, observe=True):
        if type(observe) is not bool:
            raise TypeError('observe must be boolean')
        if self._cache is None:
            raise RuntimeError('Call reset before step')
        if self._steps >= self._max_steps:
            raise RuntimeError('Interactive episode reached max_steps')
        name = str(action).strip().upper()
        if name not in self.spec.actions:
            raise ValueError(f'Unsupported interactive action {action!r}')
        with self._episode_update():
            from sampling import denoise_few_step_block
            block = self.spec.block_frames
            start = self._steps * block + self._prefix_frames - self._generated_prefix
            self._set_action(name, start, block)
            known = (self._condition if self._steps == 0 and self._generated_prefix
                     and self.backend.interactive_clamp_first_frame else None)
            shape = (1, block, *self._condition.shape[2:])
            initial = (self._draw_noise(shape) if self._initial_noise is None else
                       self._initial_noise[:, self._steps * block:(self._steps + 1) * block].clone())
            def restore(latent):
                return latent if known is None else torch.cat((known, latent[:, self._generated_prefix:]), 1)
            schedule = self._schedule
            model_targets = getattr(schedule, 'model_targets', None)
            targets = None if model_targets is None else iter(model_targets)
            def predict(current, timestep):
                times = torch.as_tensor(timestep, device=self.device, dtype=torch.float32).expand(1, block).clone()
                if known is not None:
                    times[:, :self._generated_prefix] = 0
                target = None if targets is None else next(targets)
                flow = self._flow(restore(current), times, start, self._context, self._cache, target=target)
                if self._negative_cache is not None:
                    negative = self._flow(restore(current), times, start, self._negative_context,
                                          self._negative_cache, target=target)
                    from utils import text_cfg
                    flow = text_cfg(flow, negative, self.options['guidance_scale'])
                return flow
            clean = denoise_few_step_block(
                restore(initial), schedule, predict,
                lambda current: self._draw_noise(current.shape), restore=restore,
            )
            self._commit(clean, start)
            rendered = clean[:, self._generated_prefix:] if self._steps == 0 else clean
            self._append_latent(rendered, render=observe)
            self._steps += 1
            return self.observe() if observe else None

    @property
    def steps(self):
        return self._steps

    def camera_control_state(self):
        if self.spec.control_mode != 'camera':
            return None
        return CameraControlState(self._commanded_yaw, self.rotation_degrees * self.backend.player_frame_stride)



def _position(frame, latent, model):
    return frame * (latent.shape[-2] // model.patch_size[1]) * (
        latent.shape[-1] // model.patch_size[2]
    )


def _dmd_rollout(backend, model, rows, context, scheduler, exit_index,
                 device, *, gradient_frames=None, sink_frames=None):
    """Generate self-conditioned clean chunks; grad only at the sampled exit step.

    History is committed from detached generator x0.  Earlier denoising steps
    and all chunks outside the requested gradient tail run without grad, as in
    minWM's SelfForcingTrainingPipeline.  The score recipe receives clean chunks.
    """
    from model import Action2VModel

    if not rows:
        raise ValueError("DMD rollout requires at least one packed sample")
    batch = len(rows)
    frames = backend.temporal_layout.complete_target_frames(
        rows[0]["target_frames"], prefix_frames=len(rows[0]["condition_latent"]),
    )
    prefix_frames = len(rows[0]["condition_latent"])
    frame_shape = tuple(rows[0]["latent_frame_shape"])
    if len(frame_shape) != 3 or any(type(size) is not int or size <= 0 for size in frame_shape):
        raise ValueError("DMD requires a positive latent_frame_shape [C,H,W]")
    if frames <= 0 or any(
        backend.temporal_layout.complete_target_frames(
            row["target_frames"], prefix_frames=len(row["condition_latent"]),
        ) != frames or len(row["condition_latent"]) != prefix_frames
        or tuple(row["latent_frame_shape"]) != frame_shape
        or tuple(row["condition_latent"].shape[1:]) != frame_shape
        for row in rows
    ):
        raise ValueError("DMD batch rows must share complete AR layout")
    if not 0 <= exit_index < len(scheduler.timesteps):
        raise ValueError("DMD exit step is outside the denoising schedule")
    if gradient_frames is None:
        gradient_frames = frames
    if not 0 < gradient_frames <= frames:
        raise ValueError("DMD gradient_frames must fit generated target")
    inner = backend.unwrap(model)
    if not isinstance(inner, Action2VModel):
        raise TypeError(
            f"{type(backend).__name__} needs a native dmd_rollout adapter for its "
            "cache/prefix semantics"
        )
    native = inner.model
    generated_prefix = backend.temporal_layout.generated_prefix_frames(prefix_frames)
    target_offset = prefix_frames - generated_prefix
    generated_frames = frames + generated_prefix
    dtype = next(native.parameters()).dtype
    prepared = batch_inputs([
        backend.prepare_inputs(row, frames, device, dtype=dtype) for row in rows
    ])
    target_inputs = prepared.slice(target_offset, generated_frames)
    prefix = None
    if prefix_frames and backend.prefill_condition:
        condition = torch.stack([row["condition_latent"][:prefix_frames] for row in rows])
        condition = condition.to(device=device, dtype=dtype)
        # Full-parameter training shards the patch embedder.  Project the clean
        # condition while its FSDP parameters are materialized, outside rollout
        # autograd; the resulting prefix is a stopped causal input.
        full_params = (
            FullyShardedDataParallel.summon_full_params(
                model, writeback=False, rank0_only=False,
            ) if isinstance(model, FullyShardedDataParallel) else nullcontext()
        )
        with full_params, torch.no_grad():
            prefix = inner.prepare_condition(condition, prepared.slice(0, prefix_frames))
    shape = (batch, generated_frames, *frame_shape)
    initial_noise = torch.randn(shape, device=device, dtype=dtype)
    known = (torch.stack([row['condition_latent'] for row in rows]).to(device=device, dtype=dtype)
             if generated_prefix else None)
    height, width = initial_noise.shape[-2:]
    extra_sink = backend.temporal_layout.extra_sink_frames(prefix_frames)
    cache = inner.new_cache(batch, frames + prefix_frames, height, width,
                            device, dtype, extra_sink, sink_frames=sink_frames)

    def predict(latent, timestep, inputs, start, *, commit=False, embedding=None):
        differentiable = torch.is_grad_enabled() and not commit
        history = (cache.read_only_snapshot()
                   if differentiable and native.gradient_checkpointing else cache)
        if generated_prefix and start == 0:
            latent = torch.cat((known, latent[:, generated_prefix:]), dim=1)
            if not commit:
                timestep = timestep.clone()
                timestep[:, :generated_prefix] = 0
        return model(
            latent, timestep, context,
            **({"noisy_embedding": embedding} if embedding is not None else {}),
            **inputs.forward_kwargs(), **history.forward_kwargs(cache_text=not differentiable),
            current_start=_position(start, latent, native),
            position_start=_position(start, latent, native),
            extra_sink_size=extra_sink, commit_cache=commit,
        )

    def commit(clean, inputs, start, *, embedding=None):
        with torch.no_grad():
            predict(clean.detach(), torch.full((batch, clean.shape[1]),
                    getattr(native, 'context_timestep', 0.0), device=device),
                    inputs, start, commit=True, embedding=embedding)
            cache.validate_detached()

    start = 0
    if prefix is not None:
        for length in backend.temporal_layout.prefix_blocks(prefix_frames):
            clean = prefix.latent[:, start:start + length]
            embedding = prefix.embedding[:, :, start:start + length]
            commit(clean, prepared.slice(start, length), start, embedding=embedding)
            start += length
    if start != target_offset:
        raise RuntimeError("DMD condition prefill missed a latent frame")

    generated = []
    target_start = 0
    for length in backend.temporal_layout.target_blocks(frames, prefix_frames=prefix_frames):
        absolute = target_offset + target_start
        inputs = target_inputs.slice(target_start, length)
        current = initial_noise[:, target_start:target_start + length]
        active_grad = torch.is_grad_enabled() and target_start + length > generated_frames - gradient_frames
        scheduler.reset()
        for index in range(exit_index + 1):
            timestep = scheduler.timesteps[index].float().expand(batch, length)
            with torch.set_grad_enabled(active_grad and index == exit_index):
                flow = predict(current, timestep, inputs, absolute)
                clean = scheduler.convert_model_output(flow, sample=current, index=index).to(dtype)
            if index < exit_index:
                with torch.no_grad():
                    transition_noise = torch.randn_like(current) if scheduler.stochastic_sampling else None
                    current = sampling_step(scheduler, flow.detach(), current, index, noise=transition_noise)
        commit(clean, inputs, absolute)
        generated.append(clean)
        target_start += length
    return torch.cat(generated, dim=1)[:, generated_prefix:]

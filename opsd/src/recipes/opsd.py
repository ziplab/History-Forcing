"""OPSD distillation for local autoregressive Wan backends."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import warnings

import torch

from utils import block_starts

from dataset import REVISIT_FIELD
from scheduler import sample_block_noises, sampling_step
from .base import add_causal_cache_arguments, add_common_training_arguments, validate_common_training_arguments


STUDENT_ADAPTER = "student"
TEACHER_ADAPTER = "teacher"
MSE_SPACES = ("velocity", "score", "x")


def num_supervised_denoising_steps(value):
    message = 'num_supervised_denoising_steps must be a positive integer or null'
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as error:
            raise ValueError(message) from error
    if type(value) is not int or value <= 0:
        raise ValueError(message)
    return value


def supervision_max_timestep(value):
    """Validate the deprecated threshold without reinterpreting it as a count."""
    if isinstance(value, bool):
        raise ValueError('supervision_max_timestep must be a finite nonnegative number or null')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('supervision_max_timestep must be a finite nonnegative number or null')
    return value


def _supervision_options(count, maximum):
    if count is not None and maximum is not None:
        raise ValueError(
            'num_supervised_denoising_steps and deprecated supervision_max_timestep '
            'cannot be used together'
        )
    return (
        None if count is None else num_supervised_denoising_steps(count),
        None if maximum is None else supervision_max_timestep(maximum),
    )


@dataclass(frozen=True)
class TeacherKVBranch:
    """Extra history excludes sinks, shared recent and current; native keeps policy."""

    sink_frames: int | str = 'native'
    history_frames: int | str = 'native'

    def __post_init__(self):
        for name in ('sink_frames', 'history_frames'):
            value = getattr(self, name)
            if value == 'native':
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f'teacher_kv.{name} must be a nonnegative integer or native')

    def resolve(self, native_sink, window, block_frames, recent_frames):
        sink = native_sink if self.sink_frames == 'native' else self.sink_frames
        if sink > native_sink:
            raise ValueError(f'teacher_kv.sink_frames exceeds the {native_sink} native fixed sink frames')
        history = 0 if self.history_frames == 'native' else self.history_frames
        if window != -1 and sink + recent_frames + block_frames + history > window:
            raise ValueError('teacher_kv budgets and the current block exceed the model attention window')
        return type(self)(sink, self.history_frames)

    def capacity(self, recent_frames, block_frames):
        return (None if self.history_frames == 'native' else
                self.sink_frames + self.history_frames + recent_frames + block_frames)


@dataclass(frozen=True)
class TeacherKVConfig:
    """Supervision contexts; student and fixed-camera rollouts remain native."""

    shared_recent_frames: int = 4
    positive: TeacherKVBranch = field(default_factory=TeacherKVBranch)
    negative: TeacherKVBranch = field(default_factory=lambda: TeacherKVBranch(0, 0))

    def __post_init__(self):
        if type(self.shared_recent_frames) is not int or self.shared_recent_frames < 0:
            raise ValueError('teacher_kv.shared_recent_frames must be a nonnegative integer')
        if not isinstance(self.positive, TeacherKVBranch) or not isinstance(self.negative, TeacherKVBranch):
            raise TypeError('teacher_kv branches must be TeacherKVBranch instances')

    @classmethod
    def from_value(cls, value=None):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict) or set(value) - {'shared_recent_frames', 'positive', 'negative'}:
            raise ValueError('teacher_kv expects shared_recent_frames, positive and negative mappings')
        defaults = cls()
        branches = {}
        for name in ('positive', 'negative'):
            options = value.get(name, {})
            if not isinstance(options, dict) or set(options) - {'sink_frames', 'history_frames'}:
                raise ValueError(f'teacher_kv.{name} expects sink_frames and history_frames')
            default = getattr(defaults, name)
            branches[name] = TeacherKVBranch(**{
                key: options.get(key, getattr(default, key))
                for key in ('sink_frames', 'history_frames')})
        return cls(value.get('shared_recent_frames', defaults.shared_recent_frames), **branches)

    def resolve(self, native_sink, window, block_frames, available_frames):
        recent = self.shared_recent_frames
        visible = available_frames if window == -1 else min(available_frames, window - native_sink - block_frames)
        if recent > visible:
            raise ValueError(f'teacher_kv.shared_recent_frames exceeds the {visible}-frame available visible history')
        return tuple(branch.resolve(native_sink, window, block_frames, recent)
                     for branch in (self.positive, self.negative))


def prediction_mse(student_flow, teacher_flow, sigma, mse_space):
    """Compare shared-input predictions in velocity, score or clean x0 space."""
    difference = student_flow.float() - teacher_flow.float()
    if mse_space in ("score", "x"):
        sigma = torch.as_tensor(
            sigma, device=difference.device, dtype=difference.dtype
        )
        while sigma.ndim < difference.ndim:
            sigma = sigma.unsqueeze(-1)
        # x0=x_t-sigma*v; score=-(x_t+(1-sigma)*v)/sigma.
        # The shared x_t cancels, avoiding extra clean/score tensors and
        # cancellation after rounding each estimate to the activation dtype.
        scale = sigma if mse_space == "x" else (1.0 - sigma) / sigma
        difference = difference * scale
    elif mse_space != "velocity":
        raise ValueError(
            f"mse_space must be one of {MSE_SPACES}, got {mse_space!r}"
        )
    return difference.square().mean()


def _flow(
    model, latent, timestep, context, inputs,
    cache, current_start, position_start, extra_sink_size, *, cache_text=True,
):
    """Predict conditional flow without committing the noisy KV."""
    return model(
        latent, timestep, context, **inputs.forward_kwargs(),
        **cache.forward_kwargs(cache_text=cache_text), current_start=current_start,
        position_start=position_start, extra_sink_size=extra_sink_size,
        commit_cache=False,
    )


def _cache_start(frame, height, width, model):
    inner = getattr(model, "module", model)
    return (
        frame * (height // inner.model.patch_size[1])
        * (width // inner.model.patch_size[2])
    )


@torch.no_grad()
def _commit(
    model,
    latent,
    context,
    inputs,
    caches,
    cache_frame,
    position_frame,
    extra_sink_size,
):
    batch, frames, _, height, width = latent.shape
    inner = getattr(model, 'module', model)
    context_timestep = getattr(inner.model, 'context_timestep', 0.0)
    model(
        latent,
        torch.full((batch, frames), context_timestep, device=latent.device),
        context,
        **inputs.forward_kwargs(),
        **caches.forward_kwargs(),
        current_start=_cache_start(cache_frame, height, width, model),
        position_start=_cache_start(position_frame, height, width, model),
        extra_sink_size=extra_sink_size,
        commit_cache=True,
    )


@torch.no_grad()
def _rollout(
    model,
    initial_noise,
    context,
    inputs,
    caches,
    cache_frame,
    position_frame,
    extra_sink_size,
    schedule,
    transition_noises,
    *,
    prefix=None,
):
    """Denoise and commit one native block, preserving any clean prefix slots."""
    from sampling import denoise_few_step_block

    batch, block, _, height, width = initial_noise.shape
    current_start = _cache_start(cache_frame, height, width, model)
    position_start = _cache_start(position_frame, height, width, model)
    solver = deepcopy(schedule)
    transitions = iter(transition_noises)

    def restore(current):
        return current if prefix is None else torch.cat((prefix, current[:, prefix.shape[1]:]), dim=1)

    def predict(current, timestep_value):
        timestep = timestep_value.to(torch.float32).expand(batch, block)
        if prefix is not None:
            timestep = timestep.clone()
            timestep[:, :prefix.shape[1]] = 0
        return _flow(
            model,
            current,
            timestep,
            context,
            inputs,
            caches,
            current_start,
            position_start,
            extra_sink_size,
        )
    latent = denoise_few_step_block(
        restore(initial_noise), solver, predict, lambda _: next(transitions), restore=restore,
    )
    _commit(
        model,
        latent,
        context,
        inputs,
        caches,
        cache_frame,
        position_frame,
        extra_sink_size,
    )
    return latent


class OPSDRecipe:
    """Own the teacher/student rollout, objective, EMA and adapter state."""

    name = "opsd-i2v"
    progress_description = "OPSD training"
    adapter_files = {
        STUDENT_ADAPTER: "student_adapter.pt",
        TEACHER_ADAPTER: "teacher_adapter.pt",
    }
    teacher_ema_file = "teacher_ema.pt"

    @staticmethod
    def _require_backend(backend):
        if not backend.supports_opsd:
            raise TypeError(f"{type(backend).__name__} does not implement the OPSD rollout contract")

    def add_arguments(self, parser, backend):
        self._require_backend(backend)
        backend.add_training_arguments(parser)
        add_causal_cache_arguments(parser)
        add_common_training_arguments(
            parser,
            "wan-action2v-i2v-opsd",
            text_encoder=False,
            num_samples=False,
        )
        parser.set_defaults(
            max_train_steps=100,
            log_every=1,
            save_every=40,
            lora_rank=32,
            learning_rate=1.0e-5,
            batch_size=1,
            gradient_accumulation=1,
            num_workers=0,
        )
        parser.add_argument('--sharding-strategy', default='NO_SHARD',
                            choices=('NO_SHARD', 'FULL_SHARD', 'SHARD_GRAD_OP', 'HYBRID_SHARD'),
                            help='Sharding strategy for FSDP training')
        parser.add_argument("--ema-decay", type=float, default=0.9999)
        parser.add_argument(
            "--mse-space",
            choices=MSE_SPACES,
            default="velocity",
            help=(
                "Compute teacher/student MSE in velocity, rectified-flow score, "
                "or predicted clean latent x0 space (x)"
            ),
        )
        parser.add_argument(
            "--sampling-steps",
            type=backend.sampling_steps_type,
            default=backend.sampling_steps,
            help="Denoising step count or explicit timesteps, as supported by the backend",
        )
        parser.add_argument('--timestep-shift', type=float, default=backend.timestep_shift)
        parser.add_argument('--sampling-mode', choices=('ode', 'sde'), default=None)
        parser.add_argument('--guidance-scale', type=float, default=1.0,
                            help='Native text CFG; 1 is conditional-only, 0 selects negative text. '
                                 'Independent of history guidance.')
        supervision = parser.add_mutually_exclusive_group()
        supervision.add_argument(
            '--num-supervised-denoising-steps', type=num_supervised_denoising_steps, default=None,
            help='Supervise only the last N denoising steps of each supervised chunk. '
                 'Null supervises all steps; both rollouts keep the complete sampling schedule.',
        )
        supervision.add_argument(
            '--supervision-max-timestep', type=supervision_max_timestep, default=None,
            help='Deprecated: supervise unshifted sampling timesteps <= this threshold. '
                 'Use --num-supervised-denoising-steps for last-N-step supervision.',
        )
        parser.add_argument('--history-guidance-scale',
                            dest='history_guidance_scale', type=float, default=3.0,
                            help="History guidance: target = v_local + scale * (v_history - v_local). "
                                 "v_history reads teacher history with any requested recent replacements; "
                                 "Teacher KV budgets control both branches. Scale=1 skips v_local.")
        parser.add_argument('--teacher-kv', type=TeacherKVConfig.from_value, default=TeacherKVConfig(),
                            help='Teacher supervision KV budgets as JSON, or a teacher_kv YAML mapping')

    def validate_arguments(self, parser, args):
        try:
            args.num_supervised_denoising_steps, args.supervision_max_timestep = _supervision_options(
                args.num_supervised_denoising_steps, args.supervision_max_timestep,
            )
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        if args.supervision_max_timestep is not None:
            warnings.warn(
                'supervision_max_timestep is deprecated and retains its unshifted, inclusive '
                'threshold semantics. Use num_supervised_denoising_steps for last-N-step '
                'supervision in new runs. Do not set both options.',
                FutureWarning, stacklevel=2,
            )
        if not math.isfinite(args.timestep_shift) or args.timestep_shift <= 0:
            parser.error('--timestep-shift must be finite and positive')
        validate_common_training_arguments(parser, args, num_samples=False)
        if args.lora_dropout != 0:
            parser.error('OPSD uses native eval rollouts; --lora-dropout must be 0')
        if not 0.0 <= args.ema_decay < 1.0:
            parser.error("--ema-decay must be in [0,1)")
        if not isinstance(args.sampling_steps, (int, tuple)):
            parser.error(
                "--sampling-steps must be an integer or timestep sequence"
            )
        if not math.isfinite(args.history_guidance_scale) or args.history_guidance_scale < 1.0:
            parser.error("--history-guidance-scale must be finite and >= 1; 1 is a positive-only ablation")
        if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
            parser.error('--guidance-scale must be finite and nonnegative')

    def model_config(self, backend, args):
        self._require_backend(backend)
        backend.validate_sampling_steps(args.sampling_steps)
        return backend.model_config(args)

    @staticmethod
    def dataset(path, backend):
        from dataset import OPSDLatentDataset, validate_training_dataset

        dataset = OPSDLatentDataset(path, backend.temporal_layout)
        try:
            validate_training_dataset(dataset.dataset, backend)
            return dataset
        except Exception:
            dataset.close()
            raise

    @staticmethod
    def schedule(
        device,
        backend,
        *,
        sampling_steps=None,
        timestep_shift=None,
        sampling_mode=None,
    ):
        if sampling_steps is None:
            sampling_steps = backend.sampling_steps
        return backend.opsd_schedule(device, sampling_steps, timestep_shift=timestep_shift,
                                     sampling_mode=sampling_mode)

    @staticmethod
    def prompts(sample):
        explicit = (
            sample.get("student_prompt"), sample.get("teacher_prompt")
        )
        if all(explicit):
            return explicit
        if any(explicit):
            raise ValueError(
                "OPSD student and teacher prompts must be provided together"
            )
        prompt = sample["prompt"]
        return prompt, prompt

    @staticmethod
    def _repeat_context(context, batch_size):
        if torch.is_tensor(context):
            if context.ndim != 3 or context.shape[0] != 1:
                raise ValueError(
                    "A single encoded prompt must have shape [1,L,D]"
                )
            item = context[0]
        else:
            context = tuple(context)
            if len(context) != 1:
                raise ValueError("A single prompt must produce one text context")
            item = context[0]
        return tuple(item for _ in range(batch_size))

    @staticmethod
    def encode_prompts(
        backend,
        model_dir,
        prompts,
        device,
        *,
        batch_size=1,
        guidance_scale=1.0,
    ):
        if len(prompts) != 2 or any(not prompt.strip() for prompt in prompts):
            raise ValueError("OPSD requires non-empty student and teacher prompts")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("OPSD prompt batch_size must be positive")
        prompts = backend.opsd_prompt_group(prompts, guidance_scale)
        encoder = None
        try:
            encoder = backend.text_encoder(model_dir, device)
            with torch.no_grad():
                contexts = {
                    prompt: OPSDRecipe._repeat_context(
                        backend.encode_context([prompt], device, encoder),
                        batch_size,
                    )
                    for prompt in dict.fromkeys(prompts)
                }
                return tuple(contexts[prompt] for prompt in prompts)
        finally:
            del encoder
            torch.cuda.empty_cache()

    @staticmethod
    def apply_adapters(model, rank, alpha, dropout, targets, seed, *,
                       exclude_modules=()):
        if rank == 0:
            return model
        from model import apply_opsd_lora

        return apply_opsd_lora(
            model,
            rank,
            alpha,
            dropout,
            targets,
            seed,
            STUDENT_ADAPTER,
            TEACHER_ADAPTER,
            exclude_modules=exclude_modules,
        )

    @staticmethod
    def load_initial_adapter(model, backend, checkpoint):
        backend.load_lora_weights(model, checkpoint, STUDENT_ADAPTER)
        backend.load_lora_weights(model, checkpoint, TEACHER_ADAPTER)

    def checkpoint_artifacts(
        self, model, state_dict, teacher, rank,
    ):
        if not rank:
            if teacher is None:
                raise RuntimeError("Non-LoRA OPSD requires an EMA teacher")
            return ((self.teacher_ema_file, teacher.state_dict()),)
        from model import lora_state_dict

        artifacts = [
            (
                self.adapter_files[STUDENT_ADAPTER],
                lora_state_dict(
                    model,
                    state_dict,
                    STUDENT_ADAPTER,
                ),
            ),
            (
                self.adapter_files[TEACHER_ADAPTER],
                lora_state_dict(model, state_dict, TEACHER_ADAPTER),
            ),
        ]
        return tuple(artifacts)

    def restore_artifacts(self, model, checkpoint):
        from model import load_lora_checkpoint

        for adapter, filename in self.adapter_files.items():
            load_lora_checkpoint(model, checkpoint.with_name(filename), adapter)

    @staticmethod
    def create_teacher(model, rank, decay):
        from model import FullModelEMA

        return None if rank else FullModelEMA(model, decay)

    def restore_teacher(self, teacher, checkpoint):
        if teacher is None:
            return
        path = checkpoint.with_name(self.teacher_ema_file)
        if not path.is_file():
            raise FileNotFoundError(f"Missing EMA checkpoint: {path}")
        state = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
        teacher.load_state_dict(state)

    @staticmethod
    def merge_student(state_dict, rank, alpha, device):
        from model import merge_lora_state_dict

        return merge_lora_state_dict(
            state_dict,
            rank,
            alpha,
            device,
            adapter_name=STUDENT_ADAPTER,
            discard_adapters=(TEACHER_ADAPTER,),
        )

    @staticmethod
    def update_teacher(model, teacher, decay, *, use_adapters=False):
        if teacher is not None:
            if teacher.decay != float(decay):
                raise RuntimeError("Full-model EMA decay changed during training")
            teacher.update(model)
        if use_adapters:
            from model import update_lora_ema

            update_lora_ema(
                model, decay, student=STUDENT_ADAPTER,
                teacher=TEACHER_ADAPTER,
            )

    def prepare_sample(
        self, sample, device, backend, config=None, *, batch_size=1,
    ):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("OPSD batch_size must be positive")
        return self.prepare_batch([sample] * batch_size, device, backend, config)

    def _sample_layout(self, sample, backend):
        """Validate temporal metadata before preparing or copying GPU inputs."""
        revisit = sample[REVISIT_FIELD]
        frames = len(sample["condition_latent"])
        condition_blocks = backend.condition_block_lengths(frames)
        target_frames = self.dataset_frames(sample)
        target_blocks = backend.target_block_lengths(target_frames, condition_frames=frames)
        target_starts, timeline_frames = block_starts(target_blocks)
        generated_prefix = backend.temporal_layout.generated_prefix_frames(frames)
        if timeline_frames != target_frames + generated_prefix:
            raise RuntimeError("OPSD AR block layout does not cover the target")
        if revisit + generated_prefix not in target_starts[1:]:
            raise ValueError("OPSD revisit must be an interior AR block boundary")
        return {
            "condition_block_lengths": condition_blocks,
            "generated_prefix_frames": generated_prefix,
            "extra_sink_size": backend.temporal_layout.extra_sink_frames(frames),
            "revisit": revisit + generated_prefix,
            "target_frames": timeline_frames,
            "block_lengths": target_blocks,
            "block_starts": target_starts,
        }

    def prepare_batch(self, samples, device, backend, config=None):
        """Prepare one batch, sharing unchanged controls between paired histories."""
        if not samples:
            raise ValueError('OPSD batch cannot be empty')
        layouts = [self._sample_layout(sample, backend) for sample in samples]
        first = layouts[0]
        for name in first:
            if any(layout[name] != first[name] for layout in layouts):
                raise ValueError(f'OPSD batch has different {name}; group rows by rollout layout')
        prefix_frames = sum(first['condition_block_lengths'])
        condition = torch.stack([
            sample['condition_latent'].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            for sample in samples
        ])
        prepared = {**first, 'condition': condition[:, :prefix_frames], 'initial_condition': condition,
                    'conditioning': backend.opsd_conditioning(
                        samples, self.dataset_frames(samples[0]), device)}
        return prepared

    @staticmethod
    def dataset_frames(sample):
        frames = sample.get("target_frames")
        if type(frames) is not int or frames <= 0:
            raise ValueError("OPSD data has an invalid target latent length")
        return frames

    @staticmethod
    def _caches(
        model, batch, frames, height, width, device, extra_sink_size, **options,
    ):
        inner = getattr(model, "module", model)
        return inner.new_cache(
            batch,
            frames,
            height,
            width,
            device,
            torch.bfloat16,
            extra_sink_size,
            **options,
        )

    @staticmethod
    def _set_adapter(model, adapter):
        from model import set_lora_adapter

        set_lora_adapter(model, adapter, trainable_adapter=STUDENT_ADAPTER)

    @contextmanager
    def student_parameters(self, model, use_adapters):
        """Keep Student parameters selected through the yielded loss and backward."""
        if use_adapters:
            self._set_adapter(model, STUDENT_ADAPTER)
        yield

    @contextmanager
    def teacher_parameters(self, model, teacher, use_adapters):
        """Select the Teacher adapter or scope a full-parameter EMA swap."""
        if use_adapters:
            self._set_adapter(model, TEACHER_ADAPTER)
        if teacher is None:
            yield
            return
        with teacher.average_parameters(model):
            yield

    @staticmethod
    def validate_gradients(model, use_adapters, caches):
        """Check Student gradients and detached history after loss iteration."""
        parameters = dict(model.named_parameters())
        if not use_adapters:
            trainable = [
                (name, parameter) for name, parameter in parameters.items()
                if parameter.requires_grad
            ]
            gradients = [
                (name, parameter.grad) for name, parameter in trainable
                if parameter.grad is not None
            ]
            if not trainable or not gradients:
                raise RuntimeError("Non-LoRA OPSD produced no gradients")
            invalid = [
                name for name, gradient in gradients
                if not torch.isfinite(gradient).all().item()
            ]
            if invalid:
                raise RuntimeError(
                    f"Non-finite non-LoRA OPSD gradients: {invalid[:16]}"
                )
        else:
            marker = f".{STUDENT_ADAPTER}."
            selected = [
                (name, parameter) for name, parameter in parameters.items()
                if marker in name and ".lora_" in name
            ]
            if not selected or any(
                not parameter.requires_grad for _, parameter in selected
            ):
                raise RuntimeError("OPSD student LoRA is missing or frozen")
            missing = [
                name for name, parameter in selected
                if parameter.numel() and parameter.grad is None
            ]
            if missing:
                raise RuntimeError(
                    f"Missing OPSD student LoRA gradients: {missing[:16]}"
                )
            invalid = [
                name for name, parameter in selected
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
            ]
            if invalid:
                raise RuntimeError(
                    f"Non-finite OPSD student LoRA gradients: {invalid[:16]}"
                )
            if any(
                parameter.grad is not None
                for name, parameter in parameters.items()
                if f".{TEACHER_ADAPTER}." in name
            ):
                raise RuntimeError("EMA teacher adapter received a gradient")
        for cache in caches:
            cache.validate_detached()

    @staticmethod
    def supervision_steps(schedule, num_supervised_denoising_steps=None, *, supervision_max_timestep=None):
        """Select loss steps without changing sampler timesteps or rollout execution."""
        count, maximum = _supervision_options(num_supervised_denoising_steps, supervision_max_timestep)
        total = len(schedule.timesteps)
        if total == 0:
            raise ValueError('OPSD sampling schedule must contain at least 1 denoising step')
        if maximum is not None:
            timesteps = getattr(schedule, 'unshifted_timesteps', schedule.timesteps)
            values = timesteps.detach().cpu().tolist()
            if len(values) != total:
                raise ValueError('Unshifted timesteps must match the sampling schedule length')
            selected = tuple(timestep <= maximum for timestep in values)
            if not any(selected):
                raise ValueError('supervision_max_timestep selects no sampling timesteps')
            return selected
        if count is None:
            count = total
        if count > total:
            raise ValueError(
                f'num_supervised_denoising_steps={count} exceeds the {total}-step sampling schedule'
            )
        return tuple(index >= total - count for index in range(total))

    @classmethod
    def loss_term_count(cls, sample, schedule, num_supervised_denoising_steps=None, *,
                        supervision_max_timestep=None):
        supervised_blocks = sum(
            start >= sample["revisit"] for start in sample["block_starts"]
        )
        return supervised_blocks * sum(cls.supervision_steps(
            schedule, num_supervised_denoising_steps, supervision_max_timestep=supervision_max_timestep,
        ))

    @staticmethod
    def validate_loss_options(mse_space, guidance_scale):
        if mse_space not in MSE_SPACES:
            raise ValueError(f"mse_space must be one of {MSE_SPACES}, got {mse_space!r}")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("history_guidance_scale must be finite and nonnegative")

    @staticmethod
    def teacher_target(positive_prediction, local_prediction, guidance_scale):
        """Guide the shared input toward fixed history; scale one omits the local branch."""
        if local_prediction is None:
            return positive_prediction
        local = local_prediction.float()
        return local + guidance_scale * (positive_prediction.float() - local)

    @torch.inference_mode(False)
    def loss_terms(
        self,
        model,
        teacher,
        sample,
        contexts,
        schedule,
        seed,
        *,
        mse_space="velocity",
        use_adapters=False,
        validate=False,
        sink_frames=None,
        num_supervised_denoising_steps=None,
        supervision_max_timestep=None,
        history_guidance_scale=3.0,
        teacher_kv=None,
    ):
        """Distill with normal rollout/KV tensors and caller modes restored at each yield."""
        if len(contexts) != 2:
            raise ValueError("OPSD requires student and teacher contexts")
        student_context, teacher_context = contexts
        self.validate_loss_options(mse_space, history_guidance_scale)
        supervised_steps = self.supervision_steps(
            schedule, num_supervised_denoising_steps, supervision_max_timestep=supervision_max_timestep,
        )
        teacher_kv = TeacherKVConfig.from_value(teacher_kv)
        recent_frames = teacher_kv.shared_recent_frames
        if type(use_adapters) is not bool:
            raise TypeError("use_adapters must be boolean")
        condition = sample["condition"]
        conditioning = sample["conditioning"]
        revisit = sample["revisit"]
        target_frames = sample["target_frames"]
        timeline = tuple(zip(
            sample["block_starts"], sample["block_lengths"], strict=True
        ))
        batch, _, channels, height, width = condition.shape
        condition_frames = condition.shape[1]
        extra_sink_size = sample["extra_sink_size"]
        inner = getattr(model, 'module', model).model
        native_sink = getattr(inner, 'sink_size', 0) + extra_sink_size
        block_size = max(sample['block_lengths'])
        window = -1 if inner.local_attn_size == -1 else inner.local_attn_size + extra_sink_size
        positive, negative = teacher_kv.resolve(native_sink, window, block_size, revisit)

        def new_cache(**options):
            return self._caches(
                model,
                batch,
                condition_frames + target_frames,
                height,
                width,
                condition.device,
                extra_sink_size,
                **options,
            )

        student_cache = new_cache(sink_frames=sink_frames)
        teacher_cache = new_cache()
        # Separate supervision views never commit into the fixed rollout's KV.
        teacher_revisit_cache = new_cache(capacity_frames=positive.capacity(recent_frames, block_size))
        recent_local_cache = (
            new_cache(capacity_frames=negative.capacity(recent_frames, block_size))
            if history_guidance_scale != 1.0 else None)
        tokens_per_frame = _cache_start(1, height, width, model)
        # Native sampling keeps dropout disabled; supervision enables autograd
        # independently below, including checkpointed Student forwards.
        model.eval()
        with torch.no_grad():
            condition_starts, _ = block_starts(sample["condition_block_lengths"])

            def commit_condition(context, cache):
                for start, condition_block in zip(condition_starts, sample["condition_block_lengths"], strict=True):
                    stop = start + condition_block
                    condition_latent = condition[:, start:stop]
                    _commit(
                        model,
                        condition_latent,
                        context,
                        conditioning.prefix.slice(start, condition_block),
                        cache,
                        start,
                        start,
                        extra_sink_size,
                    )

            with self.student_parameters(model, use_adapters):
                commit_condition(
                    student_context,
                    student_cache,
                )
            with self.teacher_parameters(model, teacher, use_adapters):
                commit_condition(
                    teacher_context,
                    teacher_cache,
                )

        student_recent = condition[:, max(0, condition_frames-recent_frames):].detach()
        def remember_recent(clean):
            nonlocal student_recent
            if not recent_frames:
                return
            student_recent = torch.cat((student_recent, clean.detach()), dim=1)[:, -recent_frames:]
        generator = (tuple(torch.Generator(device=condition.device).manual_seed(int(value)) for value in seed)
                     if isinstance(seed, (list, tuple)) else
                     torch.Generator(device=condition.device).manual_seed(seed))
        steps = len(schedule.timesteps)
        noise_count = steps if schedule.stochastic_sampling else 1
        for block_start, block_frames in timeline:
            if block_start >= revisit:
                break
            shape = (batch, block_frames, channels, height, width)
            initial, transitions = sample_block_noises(
                shape,
                noise_count,
                generator,
                condition.device,
                torch.bfloat16,
                stochastic_sampling=schedule.stochastic_sampling,
            )
            prefix = (sample['initial_condition']
                      if block_start == 0 and sample.get('generated_prefix_frames', 0) else None)
            with self.student_parameters(model, use_adapters):
                student_clean = _rollout(
                    model,
                    initial,
                    student_context,
                    conditioning.student.slice(block_start, block_frames),
                    student_cache,
                    condition_frames + block_start,
                    condition_frames + block_start,
                    extra_sink_size,
                    schedule,
                    transitions,
                    prefix=prefix,
                )
                remember_recent(student_clean)
            with self.teacher_parameters(model, teacher, use_adapters):
                # The rollout commits its own clean block into the fixed
                # teacher history; later blocks read it back.
                _rollout(
                    model,
                    initial,
                    teacher_context,
                    conditioning.teacher.slice(block_start, block_frames),
                    teacher_cache,
                    condition_frames + block_start,
                    condition_frames + block_start,
                    extra_sink_size,
                    schedule,
                    transitions,
                    prefix=prefix,
                )
        for block_start, block_frames in timeline:
            if block_start < revisit:
                continue
            shape = (batch, block_frames, channels, height, width)
            initial, transitions = sample_block_noises(
                shape,
                noise_count,
                generator,
                condition.device,
                torch.bfloat16,
                stochastic_sampling=schedule.stochastic_sampling,
            )
            latent = initial
            student_solver = deepcopy(schedule)
            student_solver.reset()
            block_inputs = conditioning.student.slice(block_start, block_frames)
            current_start = _cache_start(
                condition_frames + block_start, height, width, model
            )
            position_start = current_start
            recent_clean = student_recent
            recent_start = block_start - recent_frames
            def encode_recent(cache):
                with self.teacher_parameters(model, teacher, use_adapters):
                    # Keep original AR boundaries, including a partial earliest block.
                    for previous_start, previous_frames in timeline:
                        start = max(previous_start, recent_start)
                        stop = min(previous_start + previous_frames, block_start)
                        if start < stop:
                            _commit(model, recent_clean[:, start-recent_start:stop-recent_start],
                                    student_context, conditioning.student.slice(start, stop-start), cache,
                                    condition_frames + start, condition_frames + start, extra_sink_size)

            for cache, policy in ((teacher_revisit_cache, positive), (recent_local_cache, negative)):
                if cache is None:
                    continue
                cache.seed_from(teacher_cache, condition_frames + recent_start, tokens_per_frame,
                                sink_frames=policy.sink_frames,
                                history_frames=None if policy.history_frames == 'native' else policy.history_frames,
                                source_sink_frames=native_sink,
                                rebase_limit=getattr(inner, 'rope_rebase_frames', None) if recent_frames else None)
                if recent_frames:
                    encode_recent(cache)

            for index, timestep_value in enumerate(schedule.timesteps):
                timestep = timestep_value.to(torch.float32).expand(
                    batch, block_frames
                )
                positive_flow = local_flow = None
                # Both teacher branches share the noisy latent, timestep,
                # prompt, control and references; only the history differs.
                if supervised_steps[index]:
                    with self.teacher_parameters(model, teacher, use_adapters), torch.no_grad():
                        if recent_local_cache is not None:
                            local_flow = _flow(
                                model,
                                latent,
                                timestep,
                                student_context,
                                block_inputs,
                                recent_local_cache,
                                current_start,
                                position_start,
                                extra_sink_size,
                            )
                        positive_flow = _flow(
                            model,
                            latent,
                            timestep,
                            student_context,
                            block_inputs,
                            teacher_revisit_cache,
                            current_start,
                            position_start,
                            extra_sink_size,
                        )
                # Only the configured history differs. Scale=1 uses the
                # positive prediction alone.
                target = (self.teacher_target(positive_flow, local_flow, history_guidance_scale)
                          if supervised_steps[index] else None)
                # Rollout and EMA targets run without gradients; only selected
                # Student predictions build the supervised graph.
                with self.student_parameters(model, use_adapters):
                    with torch.set_grad_enabled(supervised_steps[index]):
                        student_flow = _flow(
                            model,
                            latent,
                            timestep,
                            student_context,
                            block_inputs,
                            student_cache,
                            current_start,
                            position_start,
                            extra_sink_size,
                            # Checkpoint replay must not observe a text cache
                            # mutated by the original conditional branch.
                            cache_text=False,
                        )
                        if supervised_steps[index]:
                            loss = prediction_mse(
                                student_flow,
                                target.detach(),
                                schedule.sigmas[index],
                                mse_space,
                            )
                            if (
                                not student_flow.requires_grad
                                or not loss.requires_grad
                            ):
                                trainable = sum(
                                    parameter.numel()
                                    for parameter in model.parameters()
                                    if parameter.requires_grad
                                )
                                raise RuntimeError(
                                    "OPSD student forward is detached after "
                                    "activating the student parameters "
                                    f"(trainable={trainable:,})"
                                )
                        # Keep the student adapter active until the engine has
                        # completed backward for this yielded timestep.
                        if supervised_steps[index]:
                            yield loss
                with torch.no_grad():
                    latent = sampling_step(
                        student_solver, student_flow.detach(), latent, index,
                        noise=transitions[index] if index < len(transitions) else None,
                    )
                del student_flow, positive_flow, local_flow
            with torch.no_grad():
                with self.student_parameters(model, use_adapters):
                    remember_recent(latent)
                    _commit(
                        model,
                        latent.detach(),
                        student_context,
                        block_inputs,
                        student_cache,
                        condition_frames + block_start,
                        condition_frames + block_start,
                        extra_sink_size,
                    )
                if block_start + block_frames < target_frames:
                    with self.teacher_parameters(model, teacher, use_adapters):
                        # Only later blocks read this teacher history.
                        # Advance it after supervision/backward has finished.
                        _rollout(
                            model,
                            initial,
                            teacher_context,
                            conditioning.teacher.slice(block_start, block_frames),
                            teacher_cache,
                            condition_frames + block_start,
                            condition_frames + block_start,
                            extra_sink_size,
                            schedule,
                            transitions,
                        )
        if validate:
            histories = (student_cache, teacher_cache, teacher_revisit_cache)
            if recent_local_cache is not None:
                histories += (recent_local_cache,)
            self.validate_gradients(
                model, use_adapters, histories,
            )

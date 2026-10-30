"""Video-model interface and shared Wan weights, codecs and training utilities."""

import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from wan.configs.shared import wan_shared_cfg
from model import WanModelWrapper


@dataclass(frozen=True)
class FlowOutput:
    """Model-native tensors consumed by a training recipe.

    The backend owns input preparation and the architecture-specific forward.
    The recipe owns the objective and reduction.
    """

    prediction: object
    target: object
    weight: object = None
    normalizer: object = None


@dataclass(frozen=True)
class InferenceLora:
    """An optional adapter merged into a loaded inference transformer."""

    checkpoint: Path
    targets: tuple[str, ...]
    rank: int
    alpha: int
    adapter_name: str = "default"
    dropout: float = 0.0
    seed: int = 0


class DMDGenerator(ABC):
    """Backbone operations needed to unroll a DMD generator."""

    @abstractmethod
    def dmd_layout(self, condition_frames, target_frames):
        """Return a hashable grouping key for compatible packed videos."""

    @abstractmethod
    def dmd_schedule(self, device, timesteps, *, sampling_mode="sde"):
        """Build the generator's native truncated-denoising schedule."""

    @abstractmethod
    def dmd_rollout(self, model, rows, context, scheduler, exit_index, device,
                    *, gradient_frames=None, sink_frames=None):
        """Generate from condition/control and target_frames/latent_frame_shape.

        Rows need no GT latent; gradients belong to the selected denoising step.
        """


STANDARD_PLAYER_ACTIONS = (
    "W", "A", "S", "D", "LEFT", "RIGHT", "UP", "DOWN", "WAIT",
)


@dataclass(frozen=True)
class CameraControlState:
    """Camera yaw commanded by the interactive controller, not estimated from RGB."""

    commanded_yaw_degrees: float
    rotation_degrees_per_block: float


@dataclass(frozen=True)
class InteractiveSpec:
    """Model-neutral contract advertised by a block-steppable world model."""

    actions: tuple[str, ...]
    block_frames: int
    control_mode: str
    condition_consumes_block_slot: bool = False
    control_features: int | None = None

    def __post_init__(self):
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("Interactive actions must be nonempty and unique")
        if type(self.block_frames) is not int or self.block_frames <= 0:
            raise ValueError("Interactive block_frames must be a positive integer")
        if self.control_mode not in {"camera", "discrete", "none"}:
            raise ValueError("Interactive control_mode must be camera, discrete, or none")
        if type(self.condition_consumes_block_slot) is not bool:
            raise TypeError("condition_consumes_block_slot must be boolean")
        if self.control_mode == "discrete":
            if type(self.control_features) is not int or self.control_features <= 0:
                raise ValueError("Discrete interaction requires positive control_features")
        elif self.control_features is not None:
            raise ValueError("Only discrete interaction declares control_features")


class InteractiveSession(ABC):
    """Stateful world-model episode with one observation/action boundary per step."""

    @abstractmethod
    def reset(self, sample, *, seed, max_steps):
        """Start an episode and return the initial RGB observation."""

    @abstractmethod
    def observe(self):
        """Return the latest RGB observation without advancing the model."""

    @abstractmethod
    def step(self, action, *, observe=True):
        """Advance one native block; optionally defer rendering its observation."""

    @abstractmethod
    def frames(self):
        """Return all rendered uint8 RGB frames in chronological order."""

    def camera_control_state(self) -> CameraControlState | None:
        """Return commanded camera yaw when the backend exposes its control scale."""
        return None


class LatentInteractiveSession(InteractiveSession):
    """Shared latent accumulation and causal-video rendering for model sessions."""

    def __init__(self, backend, codec):
        self.backend = backend
        self.codec = codec
        self._clear_rendering()

    def _clear_rendering(self):
        self._episode_latents = []
        self._rgb_chunks = []
        self._rgb_frames = None
        self._last_rgb_frame = None
        self._render_dirty = False
        self._stream_decoder = None
        self._color_state = {}

    def _clear_episode(self):
        self._clear_rendering()

    @contextmanager
    def _episode_update(self):
        """Discard partial generation or decoder state after a failed update."""
        try:
            yield
        except BaseException:
            self._clear_episode()
            raise

    def _start_rendering(self, latent):
        self._clear_rendering()
        with self._episode_update():
            self._episode_latents = [latent]
            self._stream_decoder = self.backend.decode_stream(self.codec)
            self._render_dirty = True
            self._render_new_frames()

    def _append_latent(self, latent, *, render=True):
        if type(render) is not bool:
            raise TypeError("render must be boolean")
        self._episode_latents.append(latent)
        self._render_dirty = True
        if render:
            self._render_new_frames()

    @torch.no_grad()
    def _render_new_frames(self):
        if not self._render_dirty:
            return
        with self._episode_update():
            streaming = self._stream_decoder is not None
            # A causal decoder needs only pending chunks; full decoders need the
            # complete episode to preserve their temporal context.
            latent = torch.cat(self._episode_latents, dim=1)
            decode = self._stream_decoder if streaming else lambda x: self.backend.decode(self.codec, x)
            video = (decode(latent) * 0.5 + 0.5).clamp(0, 1)
            if streaming:
                video = video.cpu()
            strength = getattr(self, 'options', {}).get('color_correction_strength',
                                                      getattr(self.backend, 'color_correction_strength', 0.0))
            if strength:
                from utils import chunk_color_correct
                video = chunk_color_correct(
                    video, strength, self.backend.num_frame_per_block,
                    state=self._color_state if streaming else None,
                )
            frames = (
                video[0].permute(0, 2, 3, 1).mul(255).round()
                .to(dtype=torch.uint8, device="cpu").contiguous()
            )
            if streaming:
                if len(frames) == 0:
                    raise RuntimeError("Interactive observation decode did not advance")
                # Keep rendered pixels in independent CPU chunks. Observing the latest
                # frame must not copy every preceding frame in a long episode.
                self._rgb_chunks.append(frames)
                self._rgb_frames = None
                self._episode_latents.clear()
            elif self._rgb_frames is not None and len(frames) <= len(self._rgb_frames):
                raise RuntimeError("Interactive observation decode did not advance")
            # Causal streams retain prior pixels; other codecs replace the full decode.
            # Observations and the saved video always use the same rendered timeline.
            if not streaming:
                self._rgb_frames = frames
            self._last_rgb_frame = frames[-1]
            self._render_dirty = False

    def observe(self):
        self._render_new_frames()
        if self._last_rgb_frame is None:
            raise RuntimeError("Call reset before observe")
        return self._last_rgb_frame

    def frames(self):
        self._render_new_frames()
        if self._last_rgb_frame is None:
            raise RuntimeError("Call reset before requesting video frames")
        if self._rgb_frames is None:
            self._rgb_frames = (self._rgb_chunks[0] if len(self._rgb_chunks) == 1
                                else torch.cat(self._rgb_chunks))
            self._last_rgb_frame = self._rgb_frames[-1]
            self._rgb_chunks = [self._rgb_frames]
        return self._rgb_frames


class VideoModel(ABC):
    """Own every model-specific operation used by shared entry points.

    Data loading and collation belong to the shared dataset module. Backends
    project its condition/camera/action tensors, declare sampler metadata and
    run architecture-specific forwards. Shared schedulers own sampling math.
    """

    inference_modes = ("t2v", "i2v", "v2v")
    output_includes_condition = False
    required_data_fields = ()
    requires_visual_condition_encoding = False
    supports_sft = False
    supports_opsd = False
    supports_dmd_score = False
    # Camera/action-only module paths are architecture metadata, independent
    # of the LoRA target selection. Shared visual layers are not camera modules.
    camera_modules = ()

    def freeze_camera_modules(self, model):
        """Freeze declared camera/action subtrees and return their model paths."""
        roots = [
            (name, module) for name, module in model.model.named_modules()
            if any(name == root or name.endswith("." + root)
                   for root in self.camera_modules)
        ]
        for _, module in roots:
            module.requires_grad_(False)
        return tuple(name for name, _ in roots)

    def dmd_score_forward(
        self, model, latent, timestep, context, rows, device,
    ):
        """Predict flow for DMD using this backend's normal model wrapper."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot be selected as a DMD real/fake model"
        )

    def opsd_prompt_group(self, prompts, guidance_scale=1.0):
        """Prompts required by native rollout and teacher supervision."""
        return tuple(prompts) if guidance_scale == 1.0 else (*prompts, self.negative_prompt)

    @contextmanager
    def opsd_context(self, model, contexts, guidance_scale=1.0):
        """Scope any native text guidance without changing history guidance."""
        guided = guidance_scale != 1.0
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError('guidance_scale must be finite and nonnegative')
        if len(contexts) != (3 if guided else 2):
            raise ValueError('OPSD requires student/teacher contexts and, when text CFG is enabled, a negative context')
        if not guided:
            yield contexts
            return
        with self.unwrap(model).text_guidance(contexts[2], guidance_scale):
            yield contexts[:2]

    def opsd_loss_terms(self, recipe, model, teacher, sample, contexts, *args, guidance_scale=1.0, **kwargs):
        """Let a backend with a different native sampler own its rollout."""
        with self.opsd_context(model, contexts, guidance_scale) as conditional_contexts:
            yield from self._opsd_loss_terms(recipe, model, teacher, sample, conditional_contexts, *args, **kwargs)

    def _opsd_loss_terms(self, recipe, *args, **kwargs):
        yield from recipe.loss_terms(*args, **kwargs)

    def interactive_session(
        self, model, text_encoder, codec, device, **options,
    ) -> InteractiveSession:
        """Create a stateful episode runner when the backend supports interaction."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose interactive inference"
        )

    def interactive_spec(self) -> InteractiveSpec | None:
        """Describe closed-loop support without loading weights or allocating CUDA."""
        return None

    def interactive_trajectory(self, actions, *, condition_frames=1) -> dict:
        """Convert native block actions to a full CPU condition/control timeline."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose native action trajectories"
        )

    @staticmethod
    def interactive_prompt(prompt: str) -> str:
        """Format a benchmark objective for this model's native text contract."""
        value = str(prompt).strip()
        if not value:
            raise ValueError("Interactive prompt cannot be empty")
        return value

    def interactive_input_options(self) -> dict:
        """Native manifest options for image preprocessing and default intrinsics."""
        return {}

    @staticmethod
    def checkpoint_file(checkpoint):
        """Return the single checkpoint file accepted by public entry points."""
        checkpoint = Path(checkpoint)
        if checkpoint.name != "model.pt":
            raise ValueError(
                f"Checkpoint must be named model.pt; got {checkpoint}"
            )
        return checkpoint

    @staticmethod
    def lora_checkpoint_file(checkpoint):
        """Return the adapter file accepted by training entry points."""
        checkpoint = Path(checkpoint)
        if checkpoint.name != "adapter_model.pt":
            raise ValueError(
                "LoRA checkpoint must be named adapter_model.pt; got "
                f"{checkpoint}"
            )
        return checkpoint

    def inference_lora(self, model, checkpoint):
        """Describe an optional release adapter; subclasses own only selection."""
        return None

    def merge_initial_lora(self, model, checkpoint):
        """Merge release adapters in FP32 before converting to compute precision."""
        spec = self.inference_lora(model, checkpoint)
        if spec is None:
            return model
        if not isinstance(spec, InferenceLora):
            raise TypeError("inference_lora must return InferenceLora or None")
        from model import merge_inference_lora

        original_dtype = next(model.parameters()).dtype
        model.float()
        model = merge_inference_lora(model.eval().requires_grad_(False), spec)
        return model.to(dtype=original_dtype)

    def load_initial_weights(self, model, checkpoint):
        """Shared training initialization, including a release's merged adapter."""
        self.load_weights(model, checkpoint)
        model = self.merge_initial_lora(model, checkpoint)
        return model.requires_grad_(True)


    @abstractmethod
    def validate(self, model_dir, checkpoint, *, codec=False, config=None):
        """Validate assets for inference, or the supplied backend training config."""

    @abstractmethod
    def build(self, model_dir, config):
        """Construct an unsharded model from opaque backend config."""

    @abstractmethod
    def load_weights(self, model, checkpoint):
        """Load an upstream or project checkpoint."""

    @abstractmethod
    def checkpoint_config(self, checkpoint):
        """Return project model settings, or ``None`` for upstream weights."""

    @abstractmethod
    def load(self, model_dir, checkpoint, device, dtype):
        """Construct the checkpoint's exact inference architecture."""

    @abstractmethod
    def flow_forward(self, model, batch, context, device, schedule, **options):
        """Return :class:`FlowOutput` for an architecture-specific forward."""

    @abstractmethod
    def schedule(self, device, config=None):
        """Return the model's bound FlowTrainingSchedule for SFT."""

    def inference(
        self, model, samples, text_encoder, codec, seeds, device, **options,
    ):
        """Route from the shared temporal condition; no user-specified mode.

        Subclasses override one mode hook or the common ``_inference`` path.
        Loading, collation and media handling remain outside the backend.
        """
        if not samples or len(samples) != len(seeds):
            raise ValueError("Inference requires nonempty samples and aligned seeds")
        data_length = options.get("use_data_length") or options.get(
            "num_latent_frames", options.get("num_frames"),
        ) is None
        groups = {}
        for index, sample in enumerate(samples):
            condition = sample["condition_latent"]
            frames = 0 if condition is None else len(condition)
            mode = "t2v" if frames == 0 else "i2v" if frames == 1 else "v2v"
            if mode not in self.inference_modes:
                raise ValueError(f"{type(self).__name__} does not support {mode.upper()}")
            key = (self.inference_batch_key(sample), sample.get("target_frames") if data_length else None)
            groups.setdefault((mode, key), []).append(index)

        if len(groups) == 1:
            mode = next(iter(groups))[0]
            return getattr(self, f"inference_{mode}")(
                model, samples, text_encoder, codec, seeds, device, **options,
            )

        outputs = [None] * len(samples)
        for (mode, _), indices in groups.items():
            videos, latents, images = getattr(self, f"inference_{mode}")(
                model, [samples[i] for i in indices], text_encoder, codec,
                [seeds[i] for i in indices], device, **options,
            )
            for row, index in enumerate(indices):
                outputs[index] = (videos[row], latents[row], None if images is None else images[row])
        return tuple(self._stack_inference_outputs(values) for values in zip(*outputs))

    def inference_batch_key(self, sample):
        """Group tensor-compatible inputs; override for additional model constraints."""
        condition, pixels = sample["condition_latent"], sample.get("condition_pixels")
        return (None if condition is None else tuple(condition.shape),
                None if pixels is None else tuple(pixels.shape))

    @staticmethod
    def _stack_inference_outputs(values):
        """Preserve per-sample outputs when native shapes or previews differ."""
        import torch

        if all(value is None for value in values):
            return None
        if all(value is not None and value.shape == values[0].shape for value in values):
            return torch.stack(values)
        return list(values)

    def inference_t2v(self, *args, **options):
        return self._inference(*args, **options)

    def inference_i2v(self, *args, **options):
        return self._inference(*args, **options)

    def inference_v2v(self, *args, **options):
        return self._inference(*args, **options)

    def _inference(self, *args, **options):
        raise NotImplementedError(f"{type(self).__name__} must implement its inference path")

    @abstractmethod
    def text_encoder(self, model_dir, device, *, shard_fn=None):
        """Create the model's text encoder."""

    @abstractmethod
    def encode_context(
        self, prompts, device, text_encoder, text_embeddings=None, batch_index=0,
    ):
        """Apply model-specific prompt formatting and text encoding."""

    @abstractmethod
    def codec(self, model_dir, device):
        """Create the model's video codec."""

    @abstractmethod
    def packing_assets(self, model_dir):
        """Return frozen encoding weight files without loading any models."""

    @abstractmethod
    def decode(self, codec, latent):
        """Decode a full latent timeline to video in the codec's native value range."""

    def decode_stream(self, codec):
        """Return an independent causal decoder, or use full-timeline rendering."""
        return None

    @abstractmethod
    def decode_output(self, codec, latent, condition=None):
        """Decode target-only RGB and an optional condition preview image."""

    @abstractmethod
    def decoded_frames(self, latent_frames):
        """Return the RGB frame count produced by a latent sequence."""

    @abstractmethod
    def decoded_output_frames(self, latent_frames, condition_frames=0):
        """Return target RGB frames after excluding a temporal condition."""

    def output_prefix_frames(self, condition_frames):
        """Known prefix RGB in a saved native output, excluded from fidelity metrics."""
        return self.decoded_frames(condition_frames) if (
            self.output_includes_condition and condition_frames
        ) else 0

    def validate_training_lengths(self, condition_lengths, target_lengths):
        """Validate lengths within a compatible training group, without file I/O."""

    def training_batch_key(self, condition_frames, target_frames):
        """Return a deterministic hashable key for compatible SFT microbatches."""
        return None

    def visual_encoder(self, model_dir, device):
        """Optional frozen image encoder, loaded once by the public packer."""
        return None

    def encode_visual_condition(self, codec, video, frames, *, encoder=None):
        """Encode aligned prefix RGB [1,3,P,H,W]; never read or resize media."""
        return {}

    @abstractmethod
    def transformer_layer(self):
        """Return the transformer block type used for FSDP wrapping."""

    @abstractmethod
    def apply_lora(self, model, rank, alpha, dropout, targets, seed, *,
                   freeze_camera=False):
        """Apply the model family's parameter-efficient fine-tuning."""

    @abstractmethod
    def load_lora_weights(
        self, model, checkpoint, adapter_name="default",
    ):
        """Initialize an injected trainable adapter from one checkpoint."""

    @abstractmethod
    def lora_state_dict(
        self, model, state_dict=None, adapter_name="default",
    ):
        """Export a portable PEFT adapter."""

    @abstractmethod
    def merge_lora(self, state_dict, rank, alpha, device, **options):
        """Export a normal inference state dict."""

    @abstractmethod
    def enable_gradient_checkpointing(self, model, gradient_checkpointing_func=None):
        """Enable activation checkpointing, optionally with a custom callable."""


    def add_inference_arguments(self, parser):
        return None

    def add_training_arguments(self, parser):
        return None

    def validate_training_arguments(self, parser, args):
        """Validate backend-owned training options after config parsing."""
        del parser, args

    def uses_text_encoder(self, *, training=False, config=None):
        """Whether this execution path needs a runtime text encoder."""
        del training, config
        return True

    @abstractmethod
    def model_config(self, args):
        """Validate and return serializable backend configuration."""

    @abstractmethod
    def validate_model_config(self, config):
        """Validate backend-owned settings stored in a project checkpoint."""

    def training_identity(self):
        """Native runtime policy checked by strict resume, separate from weights."""
        return {'precision': 'bf16_compute_fp32_arithmetic_v1'}

    def inference_options(self, args):
        return {}


    @staticmethod
    def unwrap(model):
        return getattr(model, "module", model)

    @staticmethod
    def normalize_state_dict(state_dict):
        from model import normalize_checkpoint_keys

        return normalize_checkpoint_keys(state_dict)

    def checkpoint(self, state_dict, *, config):
        """Serialize model weights with their backend config."""
        self.validate_model_config(config)
        return {
            "config": config,
            "model": state_dict,
        }




class WanBackend(VideoModel):
    """Own codec, text, weights and adapters without imposing an AR timeline."""

    camera_control = False
    negative_prompt = wan_shared_cfg.sample_neg_prompt
    timestep_shift = 5.0
    latent_shape = (16, 60, 104)
    pixel_shape = (480, 832)
    initial_noise_device = None

    def transformer_layer(self):
        return self.transformer_class().block_type

    @staticmethod
    def checkpoint_file(checkpoint):
        path = Path(checkpoint)
        if path.is_dir():
            for name in ("model.pt", "model.safetensors"):
                if (path / name).is_file():
                    return path / name
        return path

    def codec_config(self, model_dir):
        """Resolve the codec from latent channels, not from the Wan generation."""
        import json

        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        from wan.configs import codec_config

        return codec_config(config["out_dim"])

    def packing_assets(self, model_dir):
        return [Path(model_dir) / self.codec_config(model_dir)["vae_checkpoint"]]

    def validate(self, model_dir, checkpoint, *, codec=False, config=None):
        model_dir = Path(model_dir)
        checkpoint = self.checkpoint_file(checkpoint)
        files = [
            model_dir / "config.json",
            model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            checkpoint,
        ]
        if codec:
            files.append(model_dir / self.codec_config(model_dir)["vae_checkpoint"])
        directories = [model_dir / "google" / "umt5-xxl"]
        missing = [str(path) for path in files if not path.exists()]
        missing.extend(str(path) for path in directories if not path.is_dir())
        if missing:
            raise FileNotFoundError("Missing Wan assets:\n" + "\n".join(missing))

    def load_weights(self, model, checkpoint):
        from model import load_checkpoint, load_safetensors_weights

        checkpoint = self.checkpoint_file(checkpoint)
        if checkpoint.is_dir() or checkpoint.suffix == ".safetensors":
            return load_safetensors_weights(model.model, checkpoint)
        config = self.checkpoint_config(checkpoint)
        if config is not None:
            self.validate_model_config(config)
        return load_checkpoint(model, checkpoint)

    def checkpoint_config(self, checkpoint):
        from model import checkpoint_config

        path = self.checkpoint_file(checkpoint)
        return None if path.is_dir() or path.suffix == ".safetensors" else checkpoint_config(path)

    def load(self, model_dir, checkpoint, device, dtype):
        """Construct the architecture encoded by a checkpoint and load it."""
        stored = self.checkpoint_config(checkpoint)
        config = stored if stored is not None else {}
        model = self.build(model_dir, config)
        self.load_weights(model, checkpoint)
        return self.merge_initial_lora(model, checkpoint).to(device=device, dtype=dtype)

    def enable_gradient_checkpointing(self, model, gradient_checkpointing_func=None):
        self.unwrap(model).model.enable_gradient_checkpointing(
            gradient_checkpointing_func=gradient_checkpointing_func,
        )

    @staticmethod
    def _move(value, device):
        return value.to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )

    @classmethod
    def initial_noise(cls, seeds, shape, device, scale):
        """Draw BF16 noise and retain each sample's independent RNG stream."""
        if not seeds:
            raise ValueError("At least one inference seed is required")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Noise scale must be finite and positive")
        noise_device = cls.initial_noise_device or device
        generators = [
            torch.Generator(device=noise_device).manual_seed(int(seed))
            for seed in seeds
        ]
        noise = torch.stack([
            torch.randn(
                shape, generator=generator, device=noise_device, dtype=torch.bfloat16
            )
            for generator in generators
        ]).to(device).mul_(scale)
        return noise, generators

    @staticmethod
    def masked_flow_output(prediction, target, weight, clean, valid_mask):
        return FlowOutput(
            prediction=prediction,
            target=target,
            weight=weight * valid_mask[:, :, None, None, None],
            normalizer=valid_mask.sum(dim=1) * clean[0, 0].numel(),
        )

    def text_encoder(self, model_dir, device, *, shard_fn=None):
        from wan.t5 import T5EncoderModel

        model_dir = Path(model_dir)
        encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(model_dir / "google" / "umt5-xxl"),
            shard_fn=shard_fn,
        )
        encoder.model.eval()
        return encoder

    def encode_context(
        self, prompts, device, text_encoder, text_embeddings=None, batch_index=0,
    ):
        if text_embeddings is not None:
            return text_embeddings.encode(prompts, batch_index)
        with torch.no_grad():
            return text_encoder(prompts, device)

    def codec(self, model_dir, device):
        from wan.vae import load_vae

        config = self.codec_config(model_dir)
        return load_vae(
            vae_pth=str(Path(model_dir) / config["vae_checkpoint"]), device=device,
            z_dim=config["z_dim"],
        )

    def decode(self, codec, latent):
        from model import decode_latents

        return decode_latents(codec, latent)

    def decode_stream(self, codec):
        from model import decode_latents

        factory = getattr(codec, "decode_stream", None)
        if factory is None:
            return None
        decoder = factory()
        return lambda latent: decode_latents(codec, latent, decoder=decoder)

    def decode_output(self, codec, latent, condition=None):
        """Jointly decode history, returning target RGB and condition image."""
        if latent.ndim != 5:
            raise ValueError("Generated latent must be batched")
        if condition is not None and condition.ndim != 5:
            raise ValueError("Condition latent must be batched")
        condition_frames = 0 if condition is None else condition.shape[1]
        target_frames = self.decoded_output_frames(
            latent.shape[1], condition_frames
        )
        if condition_frames == 0:
            video, condition_image = self.decode(codec, latent), None
        else:
            if condition.shape[0] != latent.shape[0] or (
                condition.shape[2:] != latent.shape[2:]
            ):
                raise ValueError(
                    "Condition and generated latent layouts differ"
                )
            video = self.decode(
                codec, torch.cat((condition, latent), dim=1)
            )
            condition_image = video[:, 0].clone()
            prefix_frames = self.decoded_frames(condition_frames)
            video = video[:, prefix_frames:prefix_frames + target_frames]
        if (
            video.ndim != 5
            or video.shape[0] != latent.shape[0]
            or video.shape[1] != target_frames
        ):
            raise RuntimeError(
                "VAE decode produced an invalid target layout: expected "
                f"[{latent.shape[0]},{target_frames},C,H,W], got "
                f"{tuple(video.shape)}"
            )
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if condition_image is not None:
            condition_image = (
                condition_image * 0.5 + 0.5
            ).clamp(0, 1)
        return video, condition_image

    def encode(self, codec, pixels, device, dtype):
        latent = codec.encode([pixels.to(device)])[0]
        return latent.permute(1, 0, 2, 3).to(dtype)

    def inference_contexts(
        self, samples, text_encoder, device, **options,
    ):
        """Encode the positive prompt and optional sampler guidance prompt."""
        context = text_encoder(
            [sample["prompt"] for sample in samples],
            device,
        )
        return context, None

    def decoded_frames(self, latent_frames):
        from model import decoded_frame_count

        return decoded_frame_count(latent_frames)

    def decoded_output_frames(self, latent_frames, condition_frames=0):
        if latent_frames <= 0:
            raise ValueError("Target latent frame count must be positive")
        if condition_frames < 0:
            raise ValueError("Condition latent frame count cannot be negative")
        if condition_frames == 0:
            return self.decoded_frames(latent_frames)
        return (
            self.decoded_frames(condition_frames + latent_frames)
            - self.decoded_frames(condition_frames)
        )

    def apply_lora(self, model, rank, alpha, dropout, targets, seed, *,
                   freeze_camera=False):
        from model import apply_lora

        excluded = self.freeze_camera_modules(model) if freeze_camera else ()
        return apply_lora(
            model, rank, alpha, dropout, targets, seed, exclude_modules=excluded,
        )

    def load_lora_weights(
        self, model, checkpoint, adapter_name="default",
    ):
        from model import load_lora_checkpoint

        checkpoint = self.lora_checkpoint_file(checkpoint)
        return load_lora_checkpoint(
            model,
            checkpoint,
            adapter_name,
        )

    def lora_state_dict(
        self, model, state_dict=None, adapter_name="default",
    ):
        from model import lora_state_dict

        return lora_state_dict(
            model, state_dict, adapter_name
        )

    def merge_lora(self, state_dict, rank, alpha, device, **options):
        from model import merge_lora_state_dict

        return merge_lora_state_dict(
            state_dict, rank, alpha, device, **options
        )

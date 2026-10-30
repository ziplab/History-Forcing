# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Native full-context Wan policies for camera/world-model backend subclasses.

These operate on aligned RGB tensors, encoded text and latent noise. They do
not crop media, interpret camera files or add a separate inference CLI.
AR backends retain their block sampler and KV policies in wan_ar.
"""

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from wan.configs import get_config, model_config
from wan import model_class

from .base import WanBackend
from model import WanModelWrapper


@dataclass
class VideoCondition:
    """DiT condition inputs and optional clean positions preserved by the solver."""

    inputs: dict = field(default_factory=dict)
    latent: torch.Tensor | None = None
    mask: torch.Tensor | None = None

    def restore(self, sample):
        if self.mask is None:
            return sample
        return (1 - self.mask) * self.latent + self.mask * sample


class Wan:
    """Shared native UniPC loop; subclasses own conditioning and expert policy."""

    def __init__(self, models, config):
        self.models = models
        self.config = config

    @classmethod
    def transformer_class(cls, config):
        return model_class(config.base)

    @classmethod
    def model_directories(cls, config):
        return {'model': ''}

    @classmethod
    def from_pretrained(cls, model_dir, preset, *, device='cuda', dtype=torch.bfloat16):
        config = get_config(preset)
        models = {}
        for name, subfolder in cls.model_directories(config).items():
            model, info = cls.transformer_class(config).from_pretrained(
                str(Path(model_dir) / subfolder), local_files_only=True,
                torch_dtype=dtype, output_loading_info=True,
            )
            errors = {key: value for key, value in info.items() if value}
            if errors:
                raise ValueError(f'{name} checkpoint does not match the model: {errors}')
            models[name] = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        # A camera checkpoint may change patch or condition dimensions. Keep its
        # actual architecture while taking sampling defaults from the preset.
        architecture = next(iter(models.values())).config
        config.update({name: architecture[name] for name in model_config(config)
                       if name in architecture})
        return cls(models, config)

    def select_model(self, timestep, guide_scale):
        return self.models['model'], guide_scale

    def output_size(self, width, height, max_area):
        """Native I2V aspect-preserving grid; the caller also transforms intrinsics."""
        dh = self.config.vae_stride[1] * self.config.patch_size[1]
        dw = self.config.vae_stride[2] * self.config.patch_size[2]
        ratio = height / width
        output = (int(math.sqrt(max_area / ratio) // dw) * dw,
                  int(math.sqrt(max_area * ratio) // dh) * dh)
        if min(output) <= 0:
            raise ValueError('Output size is smaller than the model patch grid')
        return output

    def prepare_condition(self, video, frame_num, codec, *, clip=None):
        """Encode aligned [B,3,P,H,W] prefix RGB in [-1,1], without resizing."""
        from sampling import encode_video_condition

        if video is None:
            return VideoCondition()
        if clip is not None:
            raise ValueError('This checkpoint does not use a CLIP image encoder')
        return VideoCondition(inputs={'y': list(encode_video_condition(
            codec, video, frame_num, self.config.vae_stride,
        ))})

    def model_timesteps(self, timestep, sample, condition, seq_len):
        return timestep.expand(sample.shape[0])

    def latent_condition(self, prefix, frames, *, image_condition=None, image_embedding=None):
        """Restore native runtime inputs from previously encoded visual tensors."""
        if self.config.model_type != 'i2v':
            if prefix.shape[2]:
                raise ValueError('This text-only checkpoint does not accept a visual prefix')
            return VideoCondition()
        if not prefix.shape[2] or image_condition is None:
            raise ValueError('I2V requires a prefix and encoded image condition; repack the source RGB')
        if image_condition.shape[2] < frames:
            raise ValueError('Image condition does not cover the requested output; repack at the new length')
        return VideoCondition(inputs={'y': list(image_condition[:, :, :frames])})

    def predict(self, model, sample, timestep, context, seq_len, condition, **conditioning):
        return torch.stack(model(
            list(sample), timestep, context, seq_len,
            **condition.inputs, **conditioning,
        ))

    def sampling_shift(self, noise):
        return self.config.sample_shift

    def sampling_scheduler(self, device, steps, shift, mode):
        from scheduler import make_sampling_scheduler
        return make_sampling_scheduler(
            device, steps, solver=self.config.get('sample_solver', 'unipc'),
            timestep_shift=shift, sampling_mode=mode,
            num_train_timesteps=self.config.num_train_timesteps,
        )

    def sample(self, noise, context, negative_context, **options):
        from sampling import sample_full_context

        condition = options.pop('condition', None)
        return sample_full_context(
            self, noise, context, negative_context,
            condition=VideoCondition() if condition is None else condition, **options,
        )


class Wan21(Wan):
    """Native Wan2.1 T2V and CLIP-conditioned I2V, at any checkpoint size."""

    def prepare_condition(self, video, frame_num, codec, *, clip=None):
        result = super().prepare_condition(video, frame_num, codec)
        if video is not None:
            if clip is None:
                raise ValueError('Wan2.1 I2V requires its CLIP image encoder')
            result.inputs['clip_fea'] = clip.visual([sample[:, :1] for sample in video])
        return result

    def sampling_shift(self, noise):
        # Official Wan2.1 CLI uses shift=3 for the 480p I2V presets.
        height = noise.shape[-2] * self.config.vae_stride[1]
        width = noise.shape[-1] * self.config.vae_stride[2]
        return self.config.get('sample_shift_by_size', {}).get(
            f'{width}*{height}', self.config.sample_shift,
        )

    def latent_condition(self, prefix, frames, *, image_condition=None, image_embedding=None):
        result = super().latent_condition(
            prefix, frames, image_condition=image_condition, image_embedding=image_embedding,
        )
        if self.config.model_type == 'i2v':
            if image_embedding is None:
                raise ValueError('Wan2.1 I2V requires encoded CLIP image features')
            result.inputs['clip_fea'] = image_embedding
        return result


class Wan22A14B(Wan):
    """Native high/low-noise expert switching; distilled single DiTs do not use it."""

    @classmethod
    def model_directories(cls, config):
        return {'low_noise_model': config.low_noise_checkpoint,
                'high_noise_model': config.high_noise_checkpoint}

    def select_model(self, timestep, guide_scale):
        high = timestep.item() >= self.config.boundary * self.config.num_train_timesteps
        scale = guide_scale[int(high)] if isinstance(guide_scale, (tuple, list)) else guide_scale
        return self.models['high_noise_model' if high else 'low_noise_model'], scale


class Wan22TI2V(Wan):
    """Native 48-channel TI2V with zero timesteps and clean-latent restoration."""

    def output_size(self, width, height, max_area):
        from utils import best_output_size

        return best_output_size(
            width, height,
            self.config.vae_stride[2] * self.config.patch_size[2],
            self.config.vae_stride[1] * self.config.patch_size[1], max_area,
        )

    def prepare_condition(self, video, frame_num, codec, *, clip=None):
        if video is None:
            return VideoCondition()
        if clip is not None:
            raise ValueError('Wan2.2 TI2V does not use CLIP')
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError('Condition video must have shape [B,3,P,H,W]')
        prefix = video.shape[2]
        stride = self.config.vae_stride
        if not 0 < prefix < frame_num or (prefix - 1) % stride[0] or (frame_num - 1) % stride[0]:
            raise ValueError('RGB prefix and output must align with the causal VAE timeline')
        if video.shape[3] % stride[1] or video.shape[4] % stride[2]:
            raise ValueError('Resize RGB and transform camera intrinsics before encoding')
        encoded = torch.stack(codec.encode(list(video)))
        return self.latent_condition(encoded, (frame_num - 1) // stride[0] + 1)

    def latent_condition(self, prefix, frames, **features):
        if not prefix.shape[2]:
            return VideoCondition()
        if prefix.shape[2] >= frames:
            raise ValueError('Visual prefix must leave a nonempty generation target')
        latent = F.pad(prefix, (0, 0, 0, 0, 0, frames - prefix.shape[2]))
        mask = prefix.new_ones(prefix.shape[0], 1, frames, 1, 1)
        mask[:, :, :prefix.shape[2]] = 0
        return VideoCondition(latent=latent, mask=mask)

    def model_timesteps(self, timestep, sample, condition, seq_len):
        timestep = timestep.reshape(-1, 1).expand(sample.shape[0], 1)
        if condition.mask is None:
            return timestep.expand(sample.shape[0], seq_len)
        p, h, w = self.config.patch_size
        mask = condition.mask.expand(-1, 1, -1, sample.shape[-2], sample.shape[-1])
        times = (mask[:, 0, ::p, ::h, ::w] * timestep[:, :, None, None]).flatten(1)
        return torch.cat((times, timestep.expand(sample.shape[0], seq_len - times.shape[1])), dim=1)


class WanFullContext(WanBackend):
    """Single-DiT Wan SFT/inference backend; subclasses select a native policy.

    Camera world models override transformer_class and model_inputs. Raw media,
    spatial transforms, frozen feature encoding and LMDB remain in the packer.
    """

    output_includes_condition = True
    supports_sft = True
    supports_dmd_score = True

    @property
    @abstractmethod
    def preset(self):
        """Official preset supplying family, codec and inference defaults."""

    @property
    @abstractmethod
    def policy_class(self):
        """Wan21, Wan22TI2V or a camera-aware native policy subclass."""

    def __init__(self):
        self.config = get_config(self.preset)
        self.validate_policy()
        if self.latent_shape[0] != self.config.out_dim or any(
            pixels != latent * stride for pixels, latent, stride in zip(
                self.pixel_shape, self.latent_shape[1:], self.config.vae_stride[1:], strict=True,
            )
        ):
            raise ValueError('Declare pixel_shape and latent_shape for this backend codec')

    def validate_policy(self):
        if len(self.policy_class.model_directories(self.config)) != 1:
            raise ValueError('Use WanA14B for native high/low-noise expert training')

    def native_config(self, native):
        config = get_config(self.preset)
        config.update({key: native.config[key] for key in model_config(config) if key in native.config})
        return config

    def native_policy(self, native):
        return self.policy_class({'model': native}, self.native_config(native))

    def inference_policy(self, model):
        return self.native_policy(self.unwrap(model).model)

    @property
    def inference_modes(self):
        return {'t2v': ('t2v',), 'i2v': ('i2v', 'v2v'),
                'ti2v': ('t2v', 'i2v', 'v2v')}[self.config.model_type]

    @property
    def requires_visual_condition_encoding(self):
        return self.config.model_type == 'i2v'

    @property
    def required_data_fields(self):
        if not self.requires_visual_condition_encoding:
            return ()
        return ('image_condition_latents',) + (('image_embeddings',) if 'clip_checkpoint' in self.config else ())

    def transformer_class(self):
        return self.policy_class.transformer_class(self.config)

    def codec_config(self, model_dir):
        from wan.configs import codec_config
        return codec_config(self.config.out_dim)

    def model_config(self, args):
        return {}

    def validate_model_config(self, config):
        if config != {}:
            raise ValueError('The backend subclass and native config.json define this architecture')

    def validate_training_lengths(self, condition_lengths, target_lengths):
        for prefix in condition_lengths:
            mode = 't2v' if prefix == 0 else 'i2v' if prefix == 1 else 'v2v'
            if mode not in self.inference_modes:
                raise ValueError(f'{type(self).__name__} does not accept {mode} data')

    @staticmethod
    def checkpoint_file(checkpoint):
        path = Path(checkpoint)
        if path.is_dir() or path.suffix == '.safetensors' or path.name == 'model.pt':
            return path
        raise ValueError('Use a native model directory/safetensors or project model.pt')

    def build(self, model_dir, config):
        self.validate_model_config(config)
        native = self.transformer_class().from_config(self.transformer_class().load_config(model_dir))
        if native.model_type != self.config.model_type or native.out_dim != self.config.out_dim:
            raise ValueError('Checkpoint type/latent channels do not match the selected native backend')
        return WanModelWrapper(native)

    def load(self, model_dir, checkpoint, device, dtype):
        checkpoint = self.checkpoint_file(checkpoint)
        if not Path(checkpoint).is_dir():
            return super().load(model_dir, checkpoint, device, dtype)
        native, info = self.transformer_class().from_pretrained(
            str(checkpoint), local_files_only=True, torch_dtype=dtype, output_loading_info=True,
        )
        errors = {key: value for key, value in info.items() if value}
        if errors:
            raise ValueError(f'Native checkpoint does not match model: {errors}')
        if native.model_type != self.config.model_type or native.out_dim != self.config.out_dim:
            raise ValueError('Checkpoint type/latent channels do not match the selected native backend')
        return WanModelWrapper(native).to(device=device, dtype=dtype)

    def validate(self, model_dir, checkpoint, *, codec=False, config=None):
        root, checkpoint = Path(model_dir), self.checkpoint_file(checkpoint)
        required = [root / 'config.json', root / self.config.t5_checkpoint, root / self.config.t5_tokenizer]
        if codec:
            required.append(root / self.config.vae_checkpoint)
        if checkpoint.is_dir():
            from model import safetensors_files
            required += [checkpoint / 'config.json', *safetensors_files(checkpoint)]
        else:
            required.append(checkpoint)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Missing native Wan assets:\n' + '\n'.join(missing))

    def packing_assets(self, model_dir):
        assets = super().packing_assets(model_dir)
        if 'clip_checkpoint' in self.config:
            assets.append(Path(model_dir) / self.config.clip_checkpoint)
        return assets

    def visual_encoder(self, model_dir, device):
        if 'clip_checkpoint' not in self.config:
            return None
        from wan.wan21.clip import CLIPModel
        return CLIPModel(dtype=torch.bfloat16, device=device,
                         checkpoint_path=str(Path(model_dir) / self.config.clip_checkpoint),
                         tokenizer_path=str(Path(model_dir) / self.config.clip_tokenizer))

    @torch.no_grad()
    def encode_visual_condition(self, codec, video, frames, *, encoder=None):
        if not self.requires_visual_condition_encoding:
            return {}
        condition = self.policy_class({}, self.config).prepare_condition(
            video, self.decoded_frames(frames), codec, clip=encoder,
        )
        features = {'image_condition_latent': condition.inputs['y'][0].transpose(0, 1)}
        if 'clip_fea' in condition.inputs:
            features['image_embedding'] = condition.inputs['clip_fea'][0]
        return features

    def model_inputs(self, samples, lengths, device):
        """Batched native forward kwargs from canonical rows, shared by SFT/inference."""
        return {}

    def dmd_score_forward(
        self, model, latent, timestep, context, rows, device,
    ):
        """Run the configured full-context backend on generated DMD videos."""
        if latent.ndim != 5 or latent.shape[0] != len(rows):
            raise ValueError("DMD model needs aligned [B,F,C,H,W] videos and rows")
        if self.config.model_type != 't2v':
            videos = [torch.cat((row['condition_latent'].detach().to(sample), sample))
                      for row, sample in zip(rows, latent, strict=True)]
            output = self._conditioned_forward(model, videos, timestep, context, rows, device)
            return torch.stack([value.transpose(0, 1)[len(row['condition_latent']):]
                                for row, value in zip(rows, output, strict=True)])
        frames = latent.shape[1]
        native = self.unwrap(model).model
        seq_len = frames * math.prod(latent.shape[-2:]) // math.prod(native.patch_size)
        output = model(
            [sample.transpose(0, 1) for sample in latent], timestep, context,
            seq_len, **self.model_inputs(rows, [frames] * len(rows), device),
        )
        return torch.stack(output).transpose(1, 2)

    def schedule(self, device, config=None):
        from scheduler import make_flow_training_schedule
        return make_flow_training_schedule(device, self.config.sample_shift)

    def _condition(self, sample, frames, device):
        prefix = self._move(sample['condition_latent'], device).transpose(0, 1)[None]
        image = sample.get('image_condition_latent')
        embedding = sample.get('image_embedding')
        return self.policy_class({}, self.config).latent_condition(
            prefix, frames,
            image_condition=None if image is None else self._move(image, device).transpose(0, 1)[None],
            image_embedding=None if embedding is None else self._move(embedding, device)[None],
        )

    def _conditioned_forward(self, model, videos, timestep, context, rows, device):
        """Apply the native visual condition and token timesteps for SFT/DMD."""
        lengths = [len(video) for video in videos]
        native = self.unwrap(model).model
        policy = self.native_policy(native)
        seq_len = max(lengths) * math.prod(videos[0].shape[-2:]) // math.prod(native.patch_size)
        inputs, times, conditions = [], [], []
        for video, time, row, length in zip(videos, timestep, rows, lengths, strict=True):
            condition = self._condition(row, length, device)
            x = condition.restore(video.transpose(0, 1)[None])
            inputs.append(x[0])
            times.append(policy.model_timesteps(time, x, condition, seq_len))
            conditions.append(condition.inputs)
        merged = {}
        for name in conditions[0]:
            values = [condition[name] for condition in conditions]
            merged[name] = ([item for value in values for item in value]
                            if isinstance(values[0], list) else torch.cat(values))
        return model(inputs, torch.cat(times), context, seq_len, **merged,
                     **self.model_inputs(rows, lengths, device))

    def flow_forward(self, model, batch, context, device, schedule, **options):
        from dataset import unbatch_latent_sequences
        from scheduler import sample_flow_training
        if options.get('secondary_context') is not None:
            raise ValueError('Native full-context training requires one text prompt per video')
        clean_target = batch['clean_latent']
        samples = unbatch_latent_sequences(batch)
        lengths = [len(row['condition_latent']) + len(row['clean_latent']) for row in samples]
        frames = max(lengths)
        clean = torch.stack([F.pad(self._move(torch.cat((row['condition_latent'], row['clean_latent'])), device),
                                  (0, 0, 0, 0, 0, 0, 0, frames-length))
                             for row, length in zip(samples, lengths, strict=True)])
        noisy, target, timestep, weight = sample_flow_training(clean, frames, schedule)
        predictions, targets, weights = [], [], []
        outputs = self._conditioned_forward(
            model, [video[:length] for video, length in zip(noisy, lengths, strict=True)],
            timestep[:, 0], context, samples, device,
        )
        for i, (row, output) in enumerate(zip(samples, outputs, strict=True)):
            prefix, count = len(row['condition_latent']), len(row['clean_latent'])
            padding = (0, 0, 0, 0, 0, 0, 0, clean_target.shape[1]-count)
            predictions.append(F.pad(output.transpose(0, 1)[prefix:], padding))
            targets.append(F.pad(target[i, prefix:prefix+count], padding))
            weights.append(F.pad(weight[i, prefix:prefix+count], padding))
        return self.masked_flow_output(torch.stack(predictions), torch.stack(targets),
                                       torch.stack(weights), clean_target,
                                       batch['valid_frame_mask'].to(device))

    def add_inference_arguments(self, parser):
        parser.set_defaults(fps=self.config.sample_fps)
        parser.add_argument('--num-latent-frames', type=int, default=(self.config.frame_num-1)//4+1,
                            help='Full native output length, including the visual prefix')
        parser.add_argument('--sampling-steps', type=int, default=None)
        parser.add_argument('--sampling-mode', choices=('ode', 'sde'), default=None)
        parser.add_argument('--guidance-scale', type=self.parse_guidance_scale, default=None)
        parser.add_argument('--timestep-shift', type=float, default=None,
                            help='Default follows the native model and output resolution')
        parser.add_argument('--negative-prompt', default=None)
        parser.add_argument('--noise-scale', type=float, default=1.0)

    @staticmethod
    def parse_guidance_scale(value):
        return float(value)

    def inference_options(self, args):
        steps = self.config.sample_steps if args.sampling_steps is None else args.sampling_steps
        scale = self.config.sample_guide_scale if args.guidance_scale is None else args.guidance_scale
        negative = self.config.sample_neg_prompt if args.negative_prompt is None else args.negative_prompt
        for name in ('num_latent_frames', 'noise_scale'):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if type(steps) is not int or not 1 <= steps <= self.config.num_train_timesteps:
            raise ValueError('sampling_steps must be a valid integer step count')
        scales = scale if isinstance(scale, (tuple, list)) else [scale]
        if not scales or any(not math.isfinite(scale) for scale in scales):
            raise ValueError('guidance_scale must be finite')
        if args.timestep_shift is not None and (not math.isfinite(args.timestep_shift) or args.timestep_shift <= 0):
            raise ValueError('timestep_shift must be positive')
        return dict(num_latent_frames=args.num_latent_frames, steps=steps,
                    guide_scale=scale, shift=args.timestep_shift,
                    negative_prompt=negative, noise_scale=args.noise_scale,
                    sampling_mode=args.sampling_mode or 'ode')

    def inference_contexts(self, samples, text_encoder, device, **options):
        """Encode only the text branches read by the full-context sampler."""
        scale = options.get('guide_scale')
        scale = self.config.sample_guide_scale if scale is None else scale
        scales = scale if isinstance(scale, (tuple, list)) else (scale,)
        prompt = options.get('negative_prompt')
        prompt = self.config.sample_neg_prompt if prompt is None else prompt
        positive = (text_encoder([row['prompt'] for row in samples], device)
                    if any(scale != 0.0 for scale in scales) else [None] * len(samples))
        negative = (text_encoder([prompt] * len(samples), device)
                    if any(scale != 1.0 for scale in scales) else [None] * len(samples))
        return positive, negative

    @torch.no_grad()
    def _inference(self, model, samples, text_encoder, codec, seeds, device, **options):
        frames = options['num_latent_frames']
        contexts, negative = self.inference_contexts(samples, text_encoder, device, **options)
        policy = self.inference_policy(model)
        videos, latents, images = [], [], []
        for row, seed, context, neg in zip(samples, seeds, contexts, negative, strict=True):
            prefix = row['condition_latent']
            if frames <= len(prefix):
                raise ValueError('Output must extend beyond the visual prefix')
            shape = tuple(prefix.shape[1:])
            noise, generators = self.initial_noise([seed], (shape[0], frames, *shape[1:]), device, options['noise_scale'])
            condition = self._condition(row, frames, device)
            latent = policy.sample(noise, [context], [neg], condition=condition,
                                   steps=options['steps'], guide_scale=options['guide_scale'], shift=options['shift'],
                                   sampling_mode=options.get('sampling_mode', 'ode'),
                                   noise_generator=generators[0],
                                   **self.model_inputs([row], [frames], device))
            latent = latent.transpose(1, 2)
            video = self.decode(codec, latent).mul(.5).add(.5).clamp(0, 1)
            videos.append(video[0])
            latents.append(latent[0])
            images.append(None if len(prefix) == 0 else self.decode(
                codec, self._move(prefix, device)[None],
            )[0, 0].mul(.5).add(.5).clamp(0, 1))
        return (self._stack_inference_outputs(videos), self._stack_inference_outputs(latents),
                self._stack_inference_outputs(images))


class Wan21T2V(WanFullContext):
    """Native full-context Wan2.1 T2V; model_dir selects 1.3B or 14B."""

    preset = "wan2.1/t2v-1.3B"
    policy_class = Wan21


class WanA14B(WanFullContext):
    """Train one native noise expert; compose both experts for full-context inference.

    Subclasses declare their T2V/I2V preset and optional camera model hooks.
    The shared engine still owns FSDP, LoRA, loss, optimization and checkpointing.
    """

    policy_class = Wan22A14B

    @staticmethod
    def parse_guidance_scale(value):
        if isinstance(value, str):
            value = value.split(',')
        if not isinstance(value, (list, tuple)):
            return float(value)
        if len(value) == 1:
            return float(value[0])
        if len(value) != 2:
            raise ValueError('Use one guidance scale or LOW,HIGH')
        return tuple(float(scale) for scale in value)

    def validate_policy(self):
        names = self.policy_class.model_directories(self.config)
        if set(names) != {'low_noise_model', 'high_noise_model'}:
            raise ValueError('WanA14B requires the native low/high-noise expert layout')

    def add_training_arguments(self, parser):
        parser.add_argument('--expert', choices=tuple(self.policy_class.model_directories(self.config)), required=True,
                            help='Only this expert is loaded and trained in its native noise range')

    def model_config(self, args):
        config = {'expert': args.expert}
        self.validate_model_config(config)
        return config

    def validate_model_config(self, config):
        if (not isinstance(config, dict) or set(config) != {'expert'}
                or config['expert'] not in self.policy_class.model_directories(self.config)):
            raise ValueError('Checkpoint config must identify one native noise expert')

    @staticmethod
    def checkpoint_file(checkpoint):
        path = Path(checkpoint)
        if path.is_dir() or path.name == 'model.pt':
            return path
        raise ValueError('Use the paired expert directory or a project expert model.pt')

    def _expert_source(self, checkpoint, expert):
        path = self.checkpoint_file(checkpoint)
        if path.is_dir():
            path /= self.policy_class.model_directories(self.config)[expert]
            if (path / 'model.pt').is_file():
                path /= 'model.pt'
        if path.name == 'model.pt':
            config = self.checkpoint_config(path)
            self.validate_model_config(config)
            if config['expert'] != expert:
                raise ValueError(f'{path} contains {config["expert"]}, expected {expert}')
        return path

    def build(self, model_dir, config):
        self.validate_model_config(config)
        subfolder = self.policy_class.model_directories(self.config)[config['expert']]
        model = super().build(Path(model_dir) / subfolder, config)
        model.expert = config['expert']
        return model

    def load_weights(self, model, checkpoint):
        super().load_weights(model, self._expert_source(checkpoint, model.expert))

    def load(self, model_dir, checkpoint, device, dtype):
        """A project model.pt replaces its named expert; the other stays native.

        A paired directory may contain native shards or a project model.pt in
        each official expert subdirectory, allowing both trained experts.
        """
        checkpoint = self.checkpoint_file(checkpoint)
        stored = self.checkpoint_config(checkpoint)
        if stored is not None:
            self.validate_model_config(stored)
        sources = {
            expert: self._expert_source(
                checkpoint if stored is None or stored['expert'] == expert else model_dir, expert,
            ) for expert in self.policy_class.model_directories(self.config)
        }
        models = torch.nn.ModuleDict()
        for expert, source in sources.items():
            if source.is_dir():
                model = super().load(model_dir, source, device, dtype)
            else:
                model = self.build(model_dir, {'expert': expert}).to(device=device, dtype=dtype)
                super().load_weights(model, source)
            models[expert] = model.model
        return models

    def inference_policy(self, model):
        models = self.unwrap(model)
        return self.policy_class(models, self.native_config(next(iter(models.values()))))

    def validate(self, model_dir, checkpoint, *, codec=False, config=None):
        from model import safetensors_files

        root, checkpoint = Path(model_dir), self.checkpoint_file(checkpoint)
        required = [root / self.config.t5_checkpoint, root / self.config.t5_tokenizer]
        if codec:
            required.append(root / self.config.vae_checkpoint)
        stored = self.checkpoint_config(checkpoint)
        if stored is not None:
            self.validate_model_config(stored)
        directories = self.policy_class.model_directories(self.config)
        if config is not None:
            self.validate_model_config(config)
        for expert in directories if config is None else [config['expert']]:
            subfolder = directories[expert]
            required.append(root / subfolder / 'config.json')
            source = self._expert_source(
                root if config is None and stored is not None and stored['expert'] != expert else checkpoint, expert,
            )
            required.extend([source / 'config.json', *safetensors_files(source)] if source.is_dir() else [source])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Missing native Wan assets:\n' + '\n'.join(missing))

    def schedule(self, device, config=None):
        self.validate_model_config(config)
        schedule = super().schedule(device, config)
        high = schedule.timesteps >= self.config.boundary * self.config.num_train_timesteps
        selected = high if config['expert'] == 'high_noise_model' else ~high
        indices = selected.nonzero(as_tuple=True)[0]
        if not len(indices):
            raise ValueError('The selected expert has no training timesteps')
        return schedule.select(indices)

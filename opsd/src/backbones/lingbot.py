"""LingBot-World-Fast policy and overrides of the shared Wan backbone.

Ray-conditioned camera FiLM with shared Wan dtype and temporal inputs.
Apache-2.0; see repository LICENSE. Wan2.2 uses the shared local Wan layers.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from wan.causal import CausalWanAttentionBlock, CausalWanModel
from wan.conditioning import TemporalInputs, TemporalLayout
from wan.geometry import CameraRays

from scheduler import timestep_indices, make_lingbot_scheduler
from .wan_ar import WanAR, WanCausalModelWrapper


class CameraWanAttentionBlock(CausalWanAttentionBlock):
    def __init__(self, native, *args, **kwargs):
        super().__init__(native, *args, **kwargs)
        dim = native.self_attn.dim
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)
        for layer in (self.cam_injector_layer1, self.cam_injector_layer2,
                      self.cam_scale_layer, self.cam_shift_layer):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def inject_camera(self, x, camera_features):
        if camera_features is None:
            return x
        camera = self.cam_injector_layer2(
            F.silu(self.cam_injector_layer1(camera_features))
        )
        camera = camera + camera_features
        return (1.0 + self.cam_scale_layer(camera)) * x + self.cam_shift_layer(camera)


class CameraWanModel(CausalWanModel):
    base = "wan2.2"
    block_type = CameraWanAttentionBlock
    supported_model_types = ("t2v", "i2v")
    temporal_layout = TemporalLayout(3, image_sink=False, prefill_condition=False)
    _no_split_modules = ["CameraWanAttentionBlock"]

    def _init_conditioning(self, control_type):
        if control_type != "cam":
            raise ValueError("CameraWanModel requires control_type=cam")
        self.patch_embedding_wancamctrl = nn.Linear(
            6 * 64 * math.prod(self.patch_size), self.dim
        )
        self.c2ws_hidden_states_layer1 = nn.Linear(self.dim, self.dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(self.dim, self.dim)
        for layer in (self.patch_embedding_wancamctrl, self.c2ws_hidden_states_layer1,
                      self.c2ws_hidden_states_layer2):
            # Different rank-local video lengths invoke these layers a different
            # number of times. Keep their parameters in the enclosing FSDP unit.
            layer._fsdp_wrap = False
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _project_camera(self, camera_embedding):
        if isinstance(camera_embedding, CameraRays):
            def project(poses, intrinsics):
                packed = replace(camera_embedding, poses=poses, intrinsics=intrinsics).packed(8, dtype=torch.bfloat16)
                return self._project_camera(packed)

            chunks = []
            for start in range(0, camera_embedding.poses.shape[1], self.num_frame_per_block):
                args = (camera_embedding.poses[:, start:start+self.num_frame_per_block],
                        camera_embedding.intrinsics[:, start:start+self.num_frame_per_block])
                chunks.append(self._gradient_checkpointing_func(project, *args)
                              if torch.is_grad_enabled() and self.gradient_checkpointing else project(*args))
            return torch.cat(chunks, dim=1)
        if camera_embedding is None:
            raise ValueError("Camera-controlled forward requires camera embeddings")
        camera = rearrange(
            camera_embedding,
            "b c (f p) (h q) (w r) -> b (f h w) (c p q r)",
            p=self.patch_size[0],
            q=self.patch_size[1],
            r=self.patch_size[2],
        )
        camera = self.patch_embedding_wancamctrl(camera)
        residual = self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(camera))
        )
        return camera + residual


class LingBotWorldFast(WanAR):
    """Official I2V AR model: image/mask input channels and per-block camera FiLM."""

    camera_modules = (
        "patch_embedding_wancamctrl", "c2ws_hidden_states_layer1",
        "c2ws_hidden_states_layer2", "cam_injector_layer1",
        "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer",
    )
    inference_modes = ("i2v",)
    output_includes_condition = True
    image_condition_sink = False
    prefill_condition = False
    supports_opsd = True
    requires_visual_condition_encoding = True
    required_data_fields = ('image_condition_latents',)
    base = "wan2.2"
    num_frame_per_block = 3
    local_attn_size = 9  # Three sink frames and six recent frames.
    sink_size = 3
    camera_control = True
    player_action_full_block = True
    interactive_clamp_first_frame = False
    interactive_noise_channel_first = True
    interactive_variable_size = True
    sampling_steps_option = 'timesteps_index'
    timestep_shift_option = 'shift'
    default_rgb_frames = 81
    sampling_steps = (0, 179, 358, 679)
    sampling_steps_type = staticmethod(timestep_indices)
    timestep_shift = 10.0  # Official generate_fast CLI resolves i2v-A14B.sample_shift.

    def codec_config(self, model_dir):
        from wan.configs import codec_config

        return codec_config(16)

    @staticmethod
    def checkpoint_file(checkpoint):
        path = Path(checkpoint)
        if path.is_dir() or path.name == 'model.pt':
            return path
        raise ValueError('Use the native LingBot model directory or project model.pt')

    @staticmethod
    def _model_directory(model_dir):
        return Path(model_dir) / "lingbot_world_fast"

    def build(self, model_dir, config):
        self.validate_model_config(config)
        return WanCausalModelWrapper(CameraWanModel.from_config(
            CameraWanModel.load_config(self._model_directory(model_dir)),
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            control_type="cam",
            base=self.base,
        ))

    def load(self, model_dir, checkpoint, device, dtype):
        if not Path(checkpoint).is_dir():
            return super().load(model_dir, checkpoint, device, dtype)
        model, loading_info = CameraWanModel.from_pretrained(
            str(checkpoint),
            torch_dtype=dtype,
            local_files_only=True,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            control_type="cam",
            base=self.base,
            output_loading_info=True,
        )
        errors = {key: value for key, value in loading_info.items() if value}
        if errors:
            raise ValueError(f"LingBot-World-Fast checkpoint does not match model: {errors}")
        return WanCausalModelWrapper(model).eval().to(device)

    def validate(self, model_dir, checkpoint, *, codec=False, config=None):
        root = Path(model_dir)
        checkpoint = self.checkpoint_file(checkpoint)
        required = [
            root / "models_t5_umt5-xxl-enc-bf16.pth",
            root / "google/umt5-xxl",
            (checkpoint if checkpoint.is_dir() else self._model_directory(root)) / "config.json",
        ]
        if codec:
            required.append(root / "Wan2.1_VAE.pth")
        if checkpoint.is_dir():
            from model import safetensors_files
            required.extend(safetensors_files(checkpoint))
        else:
            required.append(checkpoint)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing LingBot assets:\n" + "\n".join(missing))


    def transformer_layer(self):
        return CameraWanAttentionBlock

    def add_length_arguments(self, parser):
        parser.add_argument(
            "--num-frames",
            type=int,
            default=self.default_rgb_frames,
            help="Requested RGB frames; native I2V rounds down to complete AR blocks",
        )
        parser.add_argument('--use-data-length', action='store_true',
                            help="Generate each packed sample's complete native timeline")

    def inference_length_options(self, args):
        if args.use_data_length:
            return {"num_frames": None}
        if args.num_frames < 9:
            raise ValueError(
                "LingBot num_frames must cover at least one three-latent block (9 RGB frames)"
            )
        return {"num_frames": args.num_frames}

    def add_inference_arguments(self, parser):
        self.add_common_inference_arguments(parser)
        parser.add_argument("--max-area", type=int, default=480 * 832)
        parser.add_argument(
            "--timestep-indices", type=timestep_indices, default=None
        )

    def inference_options(self, args):
        options = self.common_inference_options(args)
        if args.max_area < 256:
            raise ValueError("LingBot max_area must be at least 256 pixels")
        indices = timestep_indices(self.sampling_steps if args.timestep_indices is None else args.timestep_indices)
        return {
            **options,
            "max_area": args.max_area,
            "timesteps_index": indices,
            "shift": options["timestep_shift"],
        }

    def prepare_interactive_sample(self, sample, codec, device, **options):
        from sampling import encode_video_condition
        total = len(sample['viewmats']) + 1
        image, _, _ = self._inputs(
            sample, codec, device, {**options, 'num_frames': self.decoded_frames(total)},
        )
        encoded = encode_video_condition(codec, image[None, :, None], self.decoded_frames(total))[0]
        channels = self.latent_shape[0]
        return {**sample, 'condition_latent': encoded[-channels:, :1].permute(1, 0, 2, 3),
                'image_condition_latent': encoded.transpose(0, 1)}

    def validate_training_arguments(self, parser, args):
        self.checkpoint_file(args.checkpoint)

    def validate_training_lengths(self, condition_lengths, target_lengths):
        if set(condition_lengths) != {1} or min(target_lengths) < self.num_frame_per_block - 1:
            raise ValueError('LingBot requires one image prefix and a complete first generation block')

    def validate_sampling_steps(self, sampling_steps):
        if isinstance(sampling_steps, int):
            raise ValueError('LingBot sampling_steps must contain native UniPC timestep indices')
        timestep_indices(sampling_steps)

    def opsd_schedule(self, device, sampling_steps, *, timestep_shift=None, sampling_mode=None):
        return make_lingbot_scheduler(device, sampling_steps,
                               self.timestep_shift if timestep_shift is None else timestep_shift,
                               sampling_mode=self.sampling_mode if sampling_mode is None else sampling_mode)

    @torch.no_grad()
    def encode_visual_condition(self, codec, video, frames, *, encoder=None):
        from sampling import encode_video_condition
        if video.shape[2] != 1:
            raise ValueError('LingBot requires a single image condition')
        return {'image_condition_latent': encode_video_condition(codec, video, self.decoded_frames(frames))[0].transpose(0, 1)}


    @staticmethod
    def camera_condition(poses, intrinsics, height, width, device):
        from wan.geometry import compute_relative_poses
        with torch.autocast(device_type=poses.device.type, enabled=False):
            relative = torch.stack([compute_relative_poses(row, framewise=True, normalize_trans=True)
                                    for row in poses.float()])
        return CameraRays(relative.to(device), intrinsics.to(device), height, width)

    @staticmethod
    def _camera_tensors(sample, target_frames, height, width):
        """Read canonical w2c and normalized K on one continuous camera timeline."""
        views = torch.cat((sample['condition_viewmats'], sample['viewmats'][:target_frames])).float().cpu()
        k = torch.cat((sample['condition_intrinsics'], sample['intrinsics'][:target_frames])).float().cpu()
        frames = len(sample['condition_latent']) + target_frames
        if views.shape != (frames, 4, 4) or k.shape != (frames, 3, 3):
            raise ValueError('LingBot camera inputs must cover the entire generation timeline')
        ks = torch.stack((k[:, 0, 0]*width, k[:, 1, 1]*height,
                          k[:, 0, 2]*width, k[:, 1, 2]*height), dim=-1)
        return torch.linalg.inv(views), ks


    def prepare_inputs(self, sample, target_frames, device, *, dtype=torch.bfloat16):
        self.validate_training_lengths([len(sample['condition_latent'])], [target_frames])
        if 'image_condition_latent' not in sample:
            raise ValueError('LingBot needs encoded image conditions; repack the source RGB')
        frames = target_frames + 1
        height, width = sample['condition_latent'].shape[-2:]
        image = self._move(sample['image_condition_latent'][:frames], device)
        if image.shape != (frames, 20, height, width):
            raise ValueError('LingBot image conditions must cover the entire generation timeline')
        poses, ks = self._camera_tensors(sample, target_frames, height*8, width*8)
        camera = self.camera_condition(poses[None], ks[None], height*8, width*8, device)
        return TemporalInputs({'image_condition': image[None], 'camera_embedding': camera})

    def fixed_view_inputs(self, inputs, prefix_frames, target_frames):
        camera = inputs.temporal['camera_embedding']
        # Relative framewise controls keep the teacher's entire history fixed.
        teacher_poses, teacher_ks = camera.poses.clone(), camera.intrinsics.clone()
        teacher_poses[:, :prefix_frames+target_frames] = torch.eye(4, device=camera.poses.device)
        teacher_ks[:, :prefix_frames+target_frames] = camera.intrinsics[:, :1]
        return TemporalInputs({**inputs.temporal, 'camera_embedding': replace(
            camera, poses=teacher_poses, intrinsics=teacher_ks)}, inputs.static)

    def _inputs(self, sample, codec, device, options):
        condition = sample["condition_latent"]
        if condition.shape[0] != 1:
            raise ValueError("LingBot-World-Fast requires one image condition")
        if "condition_pixels" in sample:
            from utils import normalize_pixels
            image = normalize_pixels(sample["condition_pixels"]).to(device=device, dtype=torch.bfloat16)
        else:
            image = codec.decode([
                condition.permute(1, 0, 2, 3).to(device=device, dtype=codec.dtype)
            ])[0][:, 0]
        aspect = image.shape[-2] / image.shape[-1]
        area = options.get("max_area", 480 * 832)
        height = int(np.sqrt(area * aspect) // 16) * 16
        width = int(np.sqrt(area / aspect) // 16) * 16
        if min(height, width) <= 0:
            raise ValueError("LingBot resize produced an empty spatial dimension")
        image = F.interpolate(image[None].cpu(), size=(height, width), mode="bicubic")[0].to(device)
        prefix, target = sample["condition_viewmats"], sample["viewmats"]
        requested = options.get("num_frames", self.default_rgb_frames)
        requested = self.decoded_frames(len(target) + 1) if requested is None else int(requested)
        frames = (min((requested - 1) // 4 + 1, len(target) + 1) // 3) * 3
        if len(prefix) != 1 or frames < 3:
            raise ValueError("LingBot requires one condition pose and at least two target poses")
        poses, ks = self._camera_tensors(sample, frames-1, height, width)
        return image, poses, ks

    @staticmethod
    @torch.no_grad()
    def sample_image_conditioned(model, noise, context, condition, cameras, generator,
                                 *, denoise, sink_frames=None, noise_scale=1.0):
        """Bind native I2V inputs; materialize CameraRays once per AR block."""
        from sampling import sample_causal_sequence
        from wan.kv_cache.causal import make_causal_cache

        device = noise.device
        frames, height, width = noise.shape[1:]
        block = model.num_frame_per_block
        if frames % block:
            raise ValueError(f'Image-conditioned latent length must be divisible by {block}')
        cache = make_causal_cache(model, 1, frames, height, width, device, torch.bfloat16,
                                  sink_frames=sink_frames)

        def prepare_block(start, length):
            kwargs = dict(
                context=context, seq_len=length * tpf,
                image_condition=condition[None, :, start:start + length],
                camera_embedding=cameras[:, start:start + length].packed(8, dtype=torch.bfloat16),
                **cache.forward_kwargs(), current_start=start * tpf,
            )

            def predict_flow(latent, timestep):
                return model(x=latent[None], t=timestep.float().expand(1, length), **kwargs)[0]

            def commit(clean):
                predict_flow(clean, clean.new_zeros((), dtype=torch.float32))

            return predict_flow, commit

        def draw_noise(clean):
            return torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype).mul_(noise_scale)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            return sample_causal_sequence(
                noise, (block,) * (frames // block), prepare_block,
                lambda latent, predict: denoise(latent, predict, draw_noise), progress='Wan blocks',
            )

    @torch.no_grad()
    def inference_i2v(self, model, samples, text_encoder, codec, seeds, device, **options):
        from sampling import encode_video_condition, denoise_few_step_block

        if len(samples) != len(seeds):
            raise ValueError("LingBot samples and seeds must align")
        model = self.unwrap(model).model
        videos, latents, images = [], [], []
        contexts = text_encoder([sample["prompt"] for sample in samples], device)
        sampling = self.inference_schedule(device, options)
        for item, seed, context in zip(samples, seeds, contexts):
            image, poses, ks = self._inputs(item, codec, device, options)
            frames, height, width = len(poses), *image.shape[-2:]
            noises, generators = self.initial_noise(
                [seed], (16, frames, height // 8, width // 8), device, options["noise_scale"],
            )
            controls = self.camera_condition(poses[None], ks[None], height, width, device)
            condition = encode_video_condition(
                codec, image[None, :, None], (frames - 1) * 4 + 1,
            )[0]
            latent = self.sample_image_conditioned(
                model,
                noises[0],
                [context],
                condition,
                controls,
                generators[0],
                denoise=lambda latent, predict, draw: denoise_few_step_block(latent, sampling, predict, draw),

                sink_frames=options.get("sink_frames"),
                noise_scale=options["noise_scale"],
            )
            # Preserve the complete native I2V timeline, including its first slot.
            video = codec.decode([latent])[0]
            videos.append(video.permute(1, 0, 2, 3).mul(0.5).add(0.5).clamp(0, 1))
            latents.append(latent.permute(1, 0, 2, 3))
            images.append(image.mul(0.5).add(0.5).clamp(0, 1))
        return torch.stack(videos), torch.stack(latents), torch.stack(images)

from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.paths import MODEL_ROOT


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long, device=self.device)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.device)
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Optional: separate denoising schedule for the first chunk (block 0).
        # If the config does not provide `denoising_step_list_first_chunk`, all
        # blocks share `denoising_step_list` (backwards compatible).
        # This technique is proposed by [ASD](https://github.com/BigAandSmallq/SAD) for 1/2-step DMD. 
        # We thank ASD for its contribution.
        if hasattr(args, "denoising_step_list_first_chunk") and args.denoising_step_list_first_chunk is not None:
            self.denoising_step_list_first_chunk = torch.tensor(
                args.denoising_step_list_first_chunk, dtype=torch.long, device=self.device)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.device)
                self.denoising_step_list_first_chunk = timesteps[1000 - self.denoising_step_list_first_chunk]
        else:
            self.denoising_step_list_first_chunk = None

    def _initialize_models(self, args, device):
        
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.iscausal = getattr(args, "causal", True)
        model_root = getattr(args, "model_root", MODEL_ROOT)
        # Shared flow-matching shift for the score models. The generator gets it via
        # model_kwargs; real/fake must use the SAME shift so their scheduler sigmas
        # match the DMD sampling shift (self.timestep_shift) and the x0<->v
        # conversions stay consistent. Defaults to the generator's shift, else 5.0.
        self._score_timestep_shift = getattr(
            args, "timestep_shift", getattr(args, "model_kwargs", {}).get("timestep_shift", 5.0)
        )
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=self.iscausal, model_root=model_root)
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(
            model_name=self.real_model_name, is_causal=False,
            timestep_shift=self._score_timestep_shift,
            model_root=model_root,
        )
        self.real_score.model.requires_grad_(False)

        # The fake_score (critic) is a bidirectional model, matching the
        # bidirectional real_score so that the DMD gradient `v_fake - v_real` is a
        # valid same-distribution score difference. It uses the shared score shift,
        # aligned with real / generator / the DMD sampling shift.
        self.fake_score = WanDiffusionWrapper(
            model_name=self.fake_model_name, is_causal=False,
            timestep_shift=self._score_timestep_shift,
            model_root=model_root,
        )
        self.fake_score.model.requires_grad_(True)

        # Optional frozen AR teacher supplies the positive-minus-negative history
        # correction to the bidirectional teacher's target.
        self.use_history_guidance = getattr(args, "use_history_guidance", False)
        if self.use_history_guidance:
            self.ar_teacher = WanDiffusionWrapper(
                **getattr(args, "ar_teacher_kwargs", {}),
                model_name=getattr(args, "ar_teacher_name", "Wan2.1-T2V-1.3B"),
                is_causal=True,
                model_root=model_root,
            )
            self.ar_teacher.model.requires_grad_(False)

        self.text_encoder = WanTextEncoder(model_root=model_root)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_root=model_root)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep), and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # Keep the first frame's independently sampled timestep.
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent = None,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a rollout with backward simulation and select its final 21 frames.

        ``image_or_video_shape`` specifies [B, F, C, H, W]; the rollout length
        is sampled in whole blocks. ``conditional_dict`` supplies conditioning,
        and ``initial_latent`` optionally supplies a single initial frame.
        Returns the scored latents, an optional gradient mask, and the upper
        and lower denoising timesteps.
        """

        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        # Sample whole-block rollout lengths up to num_training_frames. DMD scores
        # the final 21-frame window; history guidance selects an earlier prefix
        # for each teacher branch according to its configured history budget.
        # History guidance is skipped for a single-window rollout.
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames
        self._last_generated_frames = num_generated_frames

        clean_image_or_video = None
        if clean_latent:
            clean_image_or_video = clean_latent.to(self.dtype)
            clean_image_or_video = clean_image_or_video.to(self.device)
            assert clean_image_or_video.shape == tuple(noise_shape), f"{clean_image_or_video.shape} != {tuple(noise_shape)}"

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            clean_image_or_video=clean_image_or_video,
            **conditional_dict,
        )

        # Retain the rollout for teacher history and training visualization.
        # Critic updates use only the final window and do not retain this copy.
        if not getattr(self, "_skip_segmentation", False):
            self._last_rollout_full = pred_image_or_video.to(self.dtype)
        else:
            self._last_rollout_full = None

        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Decode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Exclude the first chunk, which contains the reencoded image latent.
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        ``noise`` has shape [B, F, C, H, W]. Returns final latents
        [B, F_out, C, H, W], including the optional initial frame, followed by
        the upper and lower denoising timesteps.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,clean_image_or_video=clean_image_or_video, **conditional_dict
        )
    
    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            denoising_step_list_first_chunk=self.denoising_step_list_first_chunk,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            context_noise=self.args.context_noise
        )

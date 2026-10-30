from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
from collections.abc import Mapping
import math
import torch
import torch.distributed as dist

from model.base import SelfForcingModel
from utils.wan_wrapper import WanDiffusionWrapper
from utils.attention import validate_local_attn_size


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        Compute generator and fake-score losses from generated video samples.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        # Augment the bidirectional teacher target with the AR history difference.
        # BaseModel constructs the frozen AR teacher only when guidance is enabled.
        if getattr(self, "use_history_guidance", False):
            self.history_guidance_scale = float(getattr(args, "history_guidance_scale", 1.0))
            if not math.isfinite(self.history_guidance_scale) or self.history_guidance_scale < 0:
                raise ValueError("history_guidance_scale must be finite and non-negative")
            self.ar_frame_seq_length = getattr(args, "ar_frame_seq_length", 1560)
            self.ar_num_transformer_blocks = getattr(args, "ar_num_transformer_blocks", 30)
            self.teacher_kv = self._teacher_kv_options(args)
            # Joint mode includes the history correction in both target and normalizer.
            self.history_joint_normalization = bool(
                getattr(args, "history_joint_normalization", False))
            # Gate only the history correction, using the shifted score timestep.
            self.history_guidance_max_timestep = float(
                getattr(args, "history_guidance_max_timestep", 750))
            self.history_guidance_min_prefix_frames = getattr(
                args, "history_guidance_min_prefix_frames", 18)
            if (type(self.history_guidance_min_prefix_frames) is not int
                    or self.history_guidance_min_prefix_frames < 0):
                raise ValueError("history_guidance_min_prefix_frames must be a non-negative integer")

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _teacher_kv_options(self, args):
        """Read teacher budgets; native follows the loaded teacher's runtime policy."""
        value = getattr(args, "teacher_kv", None)
        if value is None:
            value = {}
        if not isinstance(value, Mapping) or set(value) - {"shared_recent_frames", "positive", "negative"}:
            raise ValueError("teacher_kv expects shared_recent_frames, positive and negative mappings")
        result = {"shared_recent_frames": value.get("shared_recent_frames", 6)}
        for name, history in (("positive", 12), ("negative", 0)):
            branch = value.get(name, {})
            if not isinstance(branch, Mapping) or set(branch) - {"sink_frames", "history_frames"}:
                raise ValueError(f"teacher_kv.{name} expects sink_frames and history_frames")
            result[name] = {"sink_frames": branch.get("sink_frames", 0),
                            "history_frames": branch.get("history_frames", history)}
        counts = {"shared_recent_frames": result["shared_recent_frames"]}
        counts.update({f"{name}.{field}": count for name in ("positive", "negative")
                       for field, count in result[name].items()})
        for name, count in counts.items():
            if name != "shared_recent_frames" and count == "native":
                continue
            if type(count) is not int or count < 0:
                suffix = " or native" if name != "shared_recent_frames" else ""
                raise ValueError(f"teacher_kv.{name} must be a non-negative integer{suffix}")
        return result

    def _resolve_teacher_kv(self, options, available_history=None):
        """Resolve native budgets from the teacher window, capped by real prefix length."""
        window = validate_local_attn_size(self.ar_teacher.model.local_attn_size)
        native_sink = int(getattr(self.ar_teacher.model, "sink_size", 0))
        shared = options["shared_recent_frames"]
        result = {"shared_recent_frames": shared}
        for name in ("positive", "negative"):
            branch = options[name]
            sink = native_sink if branch["sink_frames"] == "native" else branch["sink_frames"]
            history = branch["history_frames"]
            if history == "native":
                capacity = window - self.num_frame_per_block
                if available_history is not None:
                    capacity = min(capacity, available_history)
                history = capacity - sink - shared
                if history < 0:
                    raise ValueError(f"teacher_kv.{name} sink/shared history exceeds the native history capacity")
            result[name] = {"sink_frames": sink, "history_frames": history}
        return result

    @torch.no_grad()
    def _history_effect(self, student_full, x_t, timestep, conditional_dict,
                        active_mask=None):
        """Positive-minus-negative AR prediction on one scored chunk.

        Both branches independently encode the history selected by teacher_kv.
        They share the current noisy chunk, timestep, text and query RoPE positions.
        A branch with zero sink and zero total history has an empty prefix cache.
        """
        block = self.num_frame_per_block
        B, num_frames_full = student_full.shape[:2]
        seq = self.ar_frame_seq_length
        num_blocks_full = num_frames_full // block
        blocks_per_window = 21 // block
        first_window_block = num_blocks_full - blocks_per_window

        budgets = self._resolve_teacher_kv(self.teacher_kv, num_frames_full - block)
        shared = budgets["shared_recent_frames"]
        n_sink = budgets["positive"]["sink_frames"]
        n_negative_sink = budgets["negative"]["sink_frames"]
        n_recent = shared + budgets["positive"]["history_frames"]
        n_local = shared + budgets["negative"]["history_frames"]
        if n_sink < 0 or n_recent <= 0 or n_recent % block:
            raise ValueError(
                "teacher_kv.positive.sink_frames must be non-negative and its total history "
                f"a positive block multiple; got sink={n_sink}, recent={n_recent}, block={block}")
        if n_local < 0 or n_local % block or n_local >= n_recent:
            raise ValueError(
                "teacher_kv negative history must be a non-negative block multiple smaller "
                f"than positive history; got local={n_local}, recent={n_recent}, block={block}")
        window = validate_local_attn_size(self.ar_teacher.model.local_attn_size)
        if n_sink + n_recent + block > window:
            raise ValueError(
                f"Teacher KV budget {n_sink}+{n_recent}+{block} exceeds its {window}-frame window")
        if n_negative_sink + n_local + block > window:
            raise ValueError("Negative teacher_kv budget exceeds the teacher window")

        # Require enough real rollout history before the selected chunk; this
        # threshold is independent of how much history each teacher branch uses.
        valid_js = [candidate for candidate in range(1, blocks_per_window)
                    if (first_window_block + candidate) * block
                    >= self.history_guidance_min_prefix_frames]
        # Keep one synchronized RNG draw even when no chunk qualifies or the
        # timestep gate disables guidance, preserving subsequent noise draws.
        sampled_index = torch.randint(max(len(valid_js), 1), (1,), device=self.device)
        dist.broadcast(sampled_index, src=0)
        history_delta = torch.zeros_like(x_t)
        if not valid_js:
            return history_delta
        j = valid_js[sampled_index.item()]

        if active_mask is None:
            active_mask = torch.ones(B, dtype=torch.bool, device=x_t.device)
        needed = active_mask.any().to(dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(needed, op=dist.ReduceOp.MAX)
        if not needed.item():
            return history_delta

        chunk_abs_start = (first_window_block + j) * block
        rec_start = max(0, chunk_abs_start - n_recent)
        recent = student_full[:, rec_start:chunk_abs_start]
        r = recent.shape[1]
        n_local = min(n_local, r)
        # Avoid duplicating sink frames that already belong to recent history.
        sink = student_full[:, :min(n_sink, rec_start)]
        s = sink.shape[1]
        chunk_pos = s + r

        xt_blk = x_t[:, j * block:(j + 1) * block]
        tb = timestep[:, j * block:(j + 1) * block]
        history_start = chunk_pos - n_local
        local_sink = student_full[:, :min(n_negative_sink, chunk_abs_start - n_local)]
        if local_sink.shape[1] > history_start:
            raise ValueError("Negative sink does not fit the aligned teacher history")
        request = {
            "active_mask": active_mask,
            "block_size": block,
            "frame_seq_length": seq,
            "cache_dtype": self.dtype,
            "num_layers": self.ar_num_transformer_blocks,
            "chunk_position": chunk_pos,
            "branches": (
                {"sink": sink, "recent": recent,
                 "recent_start": s, "global_start": 0},
                {"sink": local_sink,
                 "recent": recent[:, -n_local:] if n_local else recent[:, :0],
                 "recent_start": history_start,
                 "global_start": 0 if local_sink.shape[1] else history_start},
            ),
        }
        # All ranks enter the same FSDP forward. Local gating happens inside its
        # parameter-owning model, after the shared weight all-gather.
        _, delta = self.ar_teacher(
            noisy_image_or_video=xt_blk, conditional_dict=conditional_dict,
            timestep=tb, history_request=request,
        )
        history_delta[:, j * block:(j + 1) * block] = delta
        return history_delta.detach()

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        student_full_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828), in x0-space.
        The fake and real scores are both taken as the model's x0 prediction, so
        grad = x0_fake - x0_real; the caller's loss is likewise in x0-space (see
        compute_distribution_matching_loss).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video (x0).
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad (x0-space).
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # in x0-space. Conditional + unconditional -> cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Add the positive-minus-negative AR history correction on a sampled chunk.
        # Each branch uses its configured history budget. Apply this correction
        # only when the rollout extends beyond the final 21-frame scoring window.
        history_delta = None
        if student_full_latent is not None and student_full_latent.shape[1] > 21:
            active_mask = None
            if self.history_guidance_max_timestep < 1e9:
                active_mask = timestep.float().mean(dim=1) < self.history_guidance_max_timestep
            history_delta = self._history_effect(
                student_full_latent, noisy_image_or_video, timestep, conditional_dict,
                active_mask=active_mask)
            if active_mask is not None:
                history_delta = history_delta * active_mask.float().view(-1, 1, 1, 1, 1)

        #   grad = (pred_fake - pred_real) / n_dmd,   n_dmd = E|x0 - pred_real|
        dims = [1, 2, 3, 4]
        eps = 1e-6
        joint_history = (
            history_delta is not None
            and getattr(self, "history_joint_normalization", False)
        )
        if joint_history:
            pred_real_target = pred_real_image + self.history_guidance_scale * history_delta
        else:
            pred_real_target = pred_real_image

        if normalization:
            n_dmd = (estimated_clean_image_or_video - pred_real_target).abs().mean(
                dim=dims, keepdim=True).detach().clamp_min(eps)
        else:
            n_dmd = torch.ones([], device=noisy_image_or_video.device)

        grad = (pred_fake_image - pred_real_target) / n_dmd

        # Norm of the pure bidirectional (DMD) term before the history correction.
        # Together with history_grad_norm this decomposes the final gradient into its two
        # sources. These are norms of separate addends (grad = bi_term - history_term),
        # so the final gradient norm need not equal their sum.
        if joint_history:
            bi_term = (pred_fake_image - pred_real_image) / n_dmd
            bi_grad_norm = torch.mean(torch.abs(bi_term)).detach()
        else:
            bi_grad_norm = torch.mean(torch.abs(grad)).detach()

        # Joint mode includes the correction in B* and the shared normalizer.
        # Otherwise subtract the unnormalized history term after normalizing DMD.
        history_grad_norm = torch.zeros([], device=noisy_image_or_video.device)
        if joint_history:
            history_term = self.history_guidance_scale * history_delta / n_dmd
            history_grad_norm = torch.mean(torch.abs(history_term)).detach()
        elif history_delta is not None:
            history_term = self.history_guidance_scale * history_delta
            grad = grad - history_term
            history_grad_norm = torch.mean(torch.abs(history_term)).detach()

        grad = torch.nan_to_num(grad)

        log_dict = {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "bi_grad_norm": bi_grad_norm,
            "history_grad_norm": history_grad_norm,
            "timestep": timestep.detach(),
        }
        return grad, log_dict

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        student_full_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]
        with torch.no_grad():

            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                student_full_latent=student_full_latent,
            )

        # The DMD loss is expressed in x0-space (original DMD recipe): the student's
        # x0 (original_latent) regresses onto the detached target (x0 - grad), where
        # grad = x0_fake - x0_real. Minimising this MSE pushes x0 along the negative
        # DMD gradient.
        if gradient_mask is not None:
            # Exclude latents masked out by the generator rollout.
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        Training uses generated videos and text prompts rather than real videos.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: unused by DMD's generated-video training path.
            - initial_latent: optional initial frame [B, 1, C, H, W].
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )
        # Use the retained rollout to build the AR teacher's history when enabled.
        student_full = getattr(self, "_last_rollout_full", None) if getattr(self, "use_history_guidance", False) else None

        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            student_full_latent=student_full,
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        Training uses generated videos and text prompts rather than real videos.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: unused by DMD's generated-video training path.
            - initial_latent: optional initial frame [B, 1, C, H, W].
        Output:
            - loss: a scalar tensor representing the critic loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # The critic trains the fake score on the standard last-21 window; AR
        # refinement is only needed for the generator's real_score, so skip it here.
        with torch.no_grad():
            self._skip_segmentation = True
            try:
                generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent
                )
            finally:
                self._skip_segmentation = False

        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict

import gc
import logging
from utils.dataset import TrainingDataIterator
from utils.dataset import TextDataset
from utils.checkpoint import TrainingCheckpoint
from trainer.evaluation import TrainingVBenchEvaluation
from trainer.performance import TrainingPerformance
from utils.distributed import EMA_FSDP, fsdp_wrap, launch_distributed_job, barrier
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os
import json


class _TrainingMetrics:
    """Sample-weighted, unscaled metrics across microbatches and ranks."""

    def __init__(self, names, device):
        self.names = tuple(names)
        self.totals = torch.zeros((2, len(self.names)), device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, log_dict, batch_size):
        for index, name in enumerate(self.names):
            if name in log_dict:
                value = log_dict[name].detach().to(dtype=self.totals.dtype).mean()
                self.totals[0, index].add_(value, alpha=batch_size)
                self.totals[1, index].add_(batch_size)

    @torch.no_grad()
    def mean(self):
        totals = self.totals.clone()
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        sums, counts = totals.cpu().tolist()
        return {name: total / count for name, total, count in zip(self.names, sums, counts)
                if count > 0}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.checkpoint = TrainingCheckpoint(config, self.device)
        self.training_evaluation = TrainingVBenchEvaluation(config, self.device)

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            # Use the configured credential, falling back to WANDB_API_KEY.
            wandb.login(
                host=config.wandb_host,
                key=getattr(config, "wandb_key", None) or os.environ.get("WANDB_API_KEY")
            )
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                entity=getattr(config, "wandb_entity", None),
                project=config.wandb_project,
                dir=config.wandb_save_dir,
                **self.checkpoint.wandb_options(),
            )

        self.output_path = config.logdir
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)

        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        # Frozen AR teacher for history-guided DMD (only when enabled; never optimized).
        if getattr(config, "use_history_guidance", False):
            self.model.ar_teacher = fsdp_wrap(
                self.model.ar_teacher,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(config, "ar_teacher_fsdp_wrap_strategy", "size"),
                cpu_offload=False
            )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if (not config.no_visualize or config.load_raw_video or config.num_training_frames > 21
                or self.training_evaluation.interval > 0):
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = TrainingDataIterator(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False) and not self.checkpoint.resuming:
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        # Load the frozen AR teacher (history-guided DMD).
        if getattr(config, "use_history_guidance", False) and getattr(config, "ar_teacher_ckpt", False):
            print(f"Loading AR teacher from {config.ar_teacher_ckpt}")
            ar_sd = torch.load(config.ar_teacher_ckpt, map_location="cpu")
            for key in ("generator", "generator_ema", "model"):
                if key in ar_sd:
                    ar_sd = ar_sd[key]
                    break
            ar_sd = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in ar_sd.items()}
            self.model.ar_teacher.load_state_dict(ar_sd, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        self.checkpoint.restore(self)
        self.performance = TrainingPerformance(config, self.device, self.model)

    def save(self):
        self.checkpoint.save(self)

    @torch.no_grad()
    def _save_latents_mp4(self, latents, path, fps=16):
        """VAE-decode clean latents [B, F, C, H, W] (first sample) and write an mp4
        (falls back to per-frame PNGs if the video writer is unavailable)."""
        import numpy as np
        video = self.model.vae.decode_to_pixel(latents.to(self.dtype))  # [B, F, C, H, W] ~[-1,1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        frames = (video[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255.0)
        frames = np.clip(frames, 0, 255).astype(np.uint8)  # [F, H, W, C]
        try:
            import imageio.v2 as imageio
            writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                        quality=8, macro_block_size=None)
            for f in frames:
                writer.append_data(f)
            writer.close()
        except Exception:
            import cv2
            os.makedirs(path + "_frames", exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(path + "_frames", f"{i:03d}.png"), f[:, :, ::-1])
        print(f"[vis] rollout ({frames.shape[0]} frames) -> {path}")

    @torch.no_grad()
    def _visualize_demo(self):
        """Generate full-length training previews with complete denoising.

        Prompts are sampled per step unless ``vis_fixed_prompts`` is enabled.
        Initial noise is seeded by prompt slot; subsequent noise uses the pipeline
        RNG. All ranks participate in the FSDP pipeline's collectives, and one
        rank per prompt writes a preview. The pipeline is initialized during
        training rollouts on all ranks.
        """
        rank = dist.get_rank()

        # Cache the prompt pool and sample with vis_seed + step. Fixed-prompt
        # previews use vis_seed alone to retain the same selection across steps.
        allp = getattr(self, "_vis_prompt_pool", None)
        if allp is None:
            path = getattr(self.config, "vis_prompt_path", "dmd/prompts/example_prompts.txt")
            with open(path) as f:
                allp = [l.strip() for l in f if l.strip()]
            self._vis_prompt_pool = allp
        k = int(getattr(self.config, "vis_num_prompts", 8))
        base_seed = int(getattr(self.config, "vis_seed", 0))
        fixed = bool(getattr(self.config, "vis_fixed_prompts", False))
        sample_seed = base_seed if fixed else base_seed + self.step
        g = torch.Generator().manual_seed(sample_seed)
        idx = torch.randperm(len(allp), generator=g)[:k].tolist()
        prompts = [allp[i] for i in idx]
        n_prompts = len(prompts)

        pipe = getattr(self.model, "inference_pipeline", None)
        if pipe is None:
            # Pipeline not built yet (only before the first training rollout). All
            # ranks see None together, so returning here keeps them in lockstep.
            return

        # Assign prompt slots by rank and seed their initial noise consistently.
        p_idx = rank % n_prompts
        my_prompt = prompts[p_idx]
        shape = list(self.config.image_or_video_shape)
        shape[0] = 1
        shape[1] = self.model.num_training_frames  # Configured rollout length.
        gen = torch.Generator(device=self.device).manual_seed(
            1234 + p_idx + int(getattr(self.config, "vis_seed", 0)))
        noise = torch.randn(shape, generator=gen, device=self.device, dtype=self.dtype)
        cond = self.model.text_encoder(text_prompts=[my_prompt])

        prev_lso = pipe.last_step_only
        pipe.last_step_only = True  # Run every denoising step for the preview.
        try:
            out = pipe.inference_with_trajectory(noise=noise, **cond)
        finally:
            pipe.last_step_only = prev_lso
        latents = out[0] if isinstance(out, tuple) else out

        # Only the FIRST rank that owns each prompt writes it (avoids duplicate writes
        # from ranks sharing a prompt); the VAE decode is local (no collectives).
        if rank < n_prompts:
            demo_dir = os.path.join(self.output_path, "demo_vis", f"prompt{p_idx}")
            os.makedirs(demo_dir, exist_ok=True)
            self._save_latents_mp4(latents, os.path.join(demo_dir, f"step{self.step:06d}.mp4"), fps=16)

        barrier()

        if self.is_main_process and not self.disable_wandb:
            video_logs = {}
            for r in range(n_prompts):
                p = os.path.join(self.output_path, "demo_vis", f"prompt{r}", f"step{self.step:06d}.mp4")
                if not os.path.exists(p):
                    continue
                try:
                    video_logs[f"demo/prompt{r}"] = wandb.Video(p, fps=16, format="mp4")
                except Exception as e:
                    print(f"[demo-vis] failed to upload {p}: {e}")
            if video_logs:
                wandb.log(video_logs, step=self.step)

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None,
                        accum_steps=1, is_last_micro=True):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            # Scale the loss so that gradients accumulated over `accum_steps`
            # micro-batches average (not sum) to one effective batch update.
            unscaled_generator_loss = generator_loss.detach()
            if accum_steps > 1:
                generator_loss = generator_loss / accum_steps
            generator_loss.backward()

            # Clip only after the final micro-batch, once all gradients are summed.
            generator_grad_norm = torch.zeros([], device=self.device)
            if is_last_micro:
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": unscaled_generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        unscaled_critic_loss = critic_loss.detach()
        if accum_steps > 1:
            critic_loss = critic_loss / accum_steps
        critic_loss.backward()

        critic_grad_norm = torch.zeros([], device=self.device)
        if is_last_micro:
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": unscaled_critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def train(self):
        start_step = self.step
        max_steps = getattr(self.config, "max_steps", None)
        # Resume an evaluation interrupted after its training checkpoint was saved.
        self.training_evaluation.run(self)

        while max_steps is None or self.step < int(max_steps):
            self.performance.start()
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            # Number of micro-batches to accumulate before each optimizer step.
            # global_batch = world_size * batch_size * accum_steps.
            accum_steps = int(getattr(self.config, "gradient_accumulate", 1) or 1)
            metric_names = ("generator_loss", "dmdtrain_gradient_norm", "history_grad_norm",
                            "bi_grad_norm", "critic_loss") if TRAIN_GENERATOR else ("critic_loss",)
            training_metrics = _TrainingMetrics(metric_names, self.device)

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)

                for i in range(accum_steps):
                    batch = next(self.dataloader)
                    self.performance.begin_microbatch(self.step, True, i, batch['prompts'])
                    generator_log_dict = self.fwdbwd_one_step(
                        batch, True,
                        accum_steps=accum_steps,
                        is_last_micro=(i == accum_steps - 1),
                    )
                    self.performance.end_microbatch(self.model)
                    training_metrics.update(generator_log_dict, len(batch["prompts"]))

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            for i in range(accum_steps):
                batch = next(self.dataloader)
                self.performance.begin_microbatch(self.step, False, i, batch['prompts'])
                critic_log_dict = self.fwdbwd_one_step(
                    batch, False,
                    accum_steps=accum_steps,
                    is_last_micro=(i == accum_steps - 1),
                )
                self.performance.end_microbatch(self.model)
                training_metrics.update(critic_log_dict, len(batch["prompts"]))

            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            performance_metrics = self.performance.finish(self.step, TRAIN_GENERATOR)

            # Generate full-length previews at the configured interval.
            vis_iters = int(getattr(self.config, "vis_iters", 0) or 0)
            if (not self.config.no_visualize) and vis_iters > 0 and self.step % vis_iters == 0:
                self._visualize_demo()

            # Logging
            # Every rank participates; only the independent logging buffer is reduced.
            wandb_loss_dict = training_metrics.mean()
            wandb_loss_dict.update(performance_metrics)
            if self.is_main_process:
                if TRAIN_GENERATOR:
                    # clip_grad_norm_ already returns the full-model norm after accumulation.
                    wandb_loss_dict["generator_grad_norm"] = generator_log_dict["generator_grad_norm"].mean().item()
                wandb_loss_dict["critic_grad_norm"] = critic_log_dict["critic_grad_norm"].mean().item()

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                metrics = {"step": self.step, "world_size": self.world_size, **wandb_loss_dict}
                print(json.dumps(metrics), flush=True)
                with open(os.path.join(self.output_path, "metrics.jsonl"), "a") as handle:
                    handle.write(json.dumps(metrics) + "\n")

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            # Publish after visualization, which also consumes CUDA RNG. A resumed
            # step must continue from the same state as this completed iteration.
            should_save = (self.step % self.config.log_iters == 0
                           or self.training_evaluation.due(self.step))
            if (not self.config.no_save) and (self.step - start_step) > 0 and should_save:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            self.training_evaluation.run(self)

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time


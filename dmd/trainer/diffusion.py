import gc
import logging

from model import CausalDiffusion
from utils.dataset import TrainingDataIterator, LatentLMDBDataset
from utils.checkpoint import TrainingCheckpoint
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
import json
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, launch_distributed_job
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.checkpoint = TrainingCheckpoint(config, self.device)

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(
                host=config.wandb_host,
                key=getattr(config, "wandb_key", None) or os.environ.get("WANDB_API_KEY")
            )
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
                **self.checkpoint.wandb_options(),
            )

        self.output_path = config.logdir
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)

        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))

        self.dataset = dataset
        # Gradient accumulation: global_batch = world_size * batch_size * accum_steps.
        self.accum_steps = int(getattr(config, "gradient_accumulate", 1) or 1)
        if self.accum_steps < 1:
            raise ValueError(f"gradient_accumulate must be >= 1, got {self.accum_steps}")
        effective_global_batch = dist.get_world_size() * int(config.batch_size) * self.accum_steps
        configured_global_batch = int(getattr(config, "total_batch_size", effective_global_batch))
        if configured_global_batch != effective_global_batch:
            raise RuntimeError(
                "AR trainer batch mismatch: "
                f"total_batch_size={configured_global_batch}, but world_size * "
                f"batch_size * gradient_accumulate={effective_global_batch}. "
                f"(world={dist.get_world_size()}, batch_size={int(config.batch_size)}, "
                f"gradient_accumulate={self.accum_steps}). Make them equal."
            )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True, seed=int(config.seed))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=int(getattr(config, "num_workers", 8)))

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
            print("DATASET LATENT SHAPE", dataset.latents_shape)
        expected_shape = tuple(int(v) for v in config.image_or_video_shape[1:])
        actual_shape = tuple(int(v) for v in dataset.latents_shape[1:])
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"LMDB/config shape mismatch: dataset row {actual_shape} != "
                f"image_or_video_shape {expected_shape}"
            )
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
            self.model.generator.load_state_dict(state_dict, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9)
        self.eval_interval = getattr(self.config, "eval_interval", 0)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        if self.eval_interval:
            self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder

        self.checkpoint.restore(self)

    def save(self):
        self.checkpoint.save(self)

    def _forward_backward(self, batch, accum_steps, is_last_micro):
        """Forward + backward for ONE micro-batch. Scales the loss by 1/accum_steps
        so gradients accumulated over accum_steps micro-batches average (not sum)
        to one effective global batch. Gradients synchronize each micro-batch.
        Returns the scaled loss for logging."""

        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)

            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        image_latent = clean_latent[:, 0:1, ]

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

        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent
        )

        # Scale so accum_steps micro-batches average to one global-batch update.
        if accum_steps > 1:
            generator_loss = generator_loss / accum_steps
        generator_loss.backward()
        return generator_loss

    def train_one_step(self, batch_unused=None):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        accum_steps = self.accum_steps
        self.generator_optimizer.zero_grad()
        generator_loss = None
        mean_loss = torch.zeros((), device=self.device)
        for i in range(accum_steps):
            batch = next(self.dataloader)
            generator_loss = self._forward_backward(
                batch, accum_steps=accum_steps,
                is_last_micro=(i == accum_steps - 1))
            mean_loss += generator_loss.detach()

        # Clip once after all micro-batches are summed, then a single opt step.
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= dist.get_world_size()

        wandb_loss_dict = {
            "generator_loss": mean_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        if self.is_main_process:
            metrics = {"step": self.step, "world_size": dist.get_world_size(), **wandb_loss_dict}
            print(json.dumps(metrics), flush=True)
            with open(os.path.join(self.output_path, "metrics.jsonl"), "a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()


    def train(self):
        configured_save_steps = getattr(self.config, "save_steps", None)
        save_steps = (
            {int(step) for step in configured_save_steps}
            if configured_save_steps else None
        )
        max_steps = int(getattr(self.config, "max_steps", 0) or 0)
        last_saved_step = self.step if self.checkpoint.resuming else -1

        while max_steps <= 0 or self.step < max_steps:
            # train_one_step pulls accum_steps micro-batches from the dataloader
            # itself, so the optimizer step corresponds to the full global batch.
            self.train_one_step()

            should_save = (
                self.step in save_steps
                if save_steps is not None
                else self.step % self.config.log_iters == 0
            )
            if (not self.config.no_save) and should_save:
                torch.cuda.empty_cache()
                self.save()
                last_saved_step = self.step
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

        if not self.config.no_save and self.step != last_saved_step:
            torch.cuda.empty_cache()
            self.save()
            torch.cuda.empty_cache()
        if self.is_main_process:
            print(f"Reached max_steps={max_steps}; AR teacher fine-tuning complete.")

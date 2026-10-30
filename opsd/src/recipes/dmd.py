"""Self Forcing distribution-matching objective, independent of backbone.

Backbones own the causal rollout and score forward contracts.  The recipe owns
the exit-step draw, generated-video score comparison, and fake-score update.
The algorithm follows Self-Forcing's rollout and alternating score updates.
Backend-specific model construction stays behind the local generator/score
contracts.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.distributed as dist

from scheduler import DEFAULT_FEW_STEP_TIMESTEPS, parse_sampling_steps
from backbones.base import DMDGenerator

from .base import add_causal_cache_arguments, add_common_training_arguments, validate_common_training_arguments


def _score_flow(backend, model, latent, timestep, context, rows, device):
    flow = backend.dmd_score_forward(model, latent, timestep, context, rows, device)
    if not torch.is_tensor(flow) or flow.shape != latent.shape:
        shape = tuple(flow.shape) if torch.is_tensor(flow) else type(flow).__name__
        raise ValueError(
            f"{type(backend).__name__} DMD score returned shape {shape}; "
            f"expected {tuple(latent.shape)}"
        )
    return flow


def _kl_gradient(fake_x0, real_x0, generated):
    difference = fake_x0.float() - real_x0.float()
    normalizer = (generated.float() - real_x0.float()).abs().flatten(1).mean(1)
    difference = difference / normalizer.clamp_min(1e-8)[:, None, None, None, None]
    # Keep upstream stabilization, but expose anomalies before it hides them.
    with torch.no_grad():
        finite = torch.isfinite(normalizer)
        metrics = {
            "train/dmd_raw_nonfinite_fraction": (~torch.isfinite(difference)).sum().double() / difference.numel(),
            "train/dmd_normalizer_nonfinite_fraction": (~finite).sum().double() / normalizer.numel(),
            "train/dmd_normalizer_finite_mean": (
                torch.where(finite, normalizer, 0).double().sum() / finite.sum().clamp_min(1)
            ),
        }
    return torch.nan_to_num(difference), metrics


def _dmd_loss(generated, gradient, gradient_frames):
    if gradient_frames <= 0 or gradient_frames > generated.shape[1]:
        raise ValueError("DMD gradient_frames must lie within the generated video")
    # Match upstream FP64 target subtraction: FP32 can erase small KL updates.
    active = generated[:, -gradient_frames:].double()
    target = active.detach() - gradient[:, -gradient_frames:].detach().double()
    return 0.5 * torch.nn.functional.mse_loss(active, target)


class DMDRecipe:
    name = "dmd"
    progress_description = "DMD"

    def add_arguments(self, parser, backend):
        if not isinstance(backend, DMDGenerator):
            raise TypeError(f"{type(backend).__name__} does not implement a few-step DMD generator")
        backend.add_training_arguments(parser)
        add_causal_cache_arguments(parser)
        add_common_training_arguments(parser, "dmd", text_encoder=False)
        parser.set_defaults(
            max_train_steps=400, batch_size=1, gradient_accumulation=1,
            save_every=100, log_every=1, lora_rank=32,
            learning_rate=2.0e-6, num_workers=0,
        )
        parser.add_argument("--real-model", required=True,
                            help="Existing backend type for the frozen real model")
        parser.add_argument("--real-model-dir", required=True, type=Path)
        parser.add_argument("--real-checkpoint", required=True, type=Path)
        parser.add_argument("--fake-model", required=True,
                            help="Existing backend type for the trainable fake model")
        parser.add_argument("--fake-model-dir", required=True, type=Path)
        parser.add_argument("--fake-checkpoint", required=True, type=Path)
        parser.add_argument("--fake-lora-rank", type=int, default=32)
        parser.add_argument("--fake-lora-alpha", type=int, default=32)
        parser.add_argument("--critic-learning-rate", type=float, default=4.0e-7)
        parser.add_argument("--generator-update-ratio", type=int, default=5)
        parser.add_argument("--sampling-steps", type=parse_sampling_steps,
                            default=DEFAULT_FEW_STEP_TIMESTEPS,
                            help="Generator denoising step count or explicit decreasing timesteps")
        parser.add_argument("--sampling-mode", choices=("ode", "sde"), default="sde")
        parser.add_argument("--last-step-only", action=argparse.BooleanOptionalAction,
                            default=False)
        parser.add_argument("--gradient-frames", type=int, default=None,
                            help="DMD generator loss and backward on final N latent frames")
        parser.add_argument("--real-guidance-scale", type=float, default=3.0)
        parser.add_argument("--fake-guidance-scale", type=float, default=0.0)
        parser.add_argument("--negative-prompt", default=backend.negative_prompt)
        parser.add_argument("--score-timestep-shift", type=float,
                            default=backend.timestep_shift)
        parser.add_argument("--ema-decay", type=float, default=0.99)
        parser.add_argument("--ema-start-step", type=int, default=200)

    def validate_arguments(self, parser, args):
        validate_common_training_arguments(parser, args)
        if args.lora_dropout != 0:
            parser.error('DMD uses eval-mode generator and score models; --lora-dropout must be 0')
        if args.fake_lora_rank < 0 or args.fake_lora_alpha <= 0:
            parser.error("Fake score LoRA rank/alpha are invalid")
        if (not math.isfinite(args.critic_learning_rate) or args.critic_learning_rate <= 0
                or args.generator_update_ratio <= 0):
            parser.error("Critic learning rate must be finite and positive and generator update ratio positive")
        if args.gradient_frames is not None and args.gradient_frames <= 0:
            parser.error("--gradient-frames must be positive")
        if not all(math.isfinite(x) and x >= 0 for x in (
            args.real_guidance_scale, args.fake_guidance_scale,
        )):
            parser.error("DMD guidance scales must be finite and nonnegative")
        if not math.isfinite(args.score_timestep_shift) or args.score_timestep_shift <= 0:
            parser.error("--score-timestep-shift must be finite and positive")
        if not 0 <= args.ema_decay < 1:
            parser.error("--ema-decay must be in [0,1)")
        if args.ema_start_step < 0:
            parser.error("--ema-start-step cannot be negative")
    @staticmethod
    def exit_step(steps, device, *, last_step_only):
        index = torch.randint(0, steps, (1,), device=device)
        if last_step_only:
            index.fill_(steps - 1)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(index, src=0)
        return int(index.item())

    @staticmethod
    def score_timestep(batch, device, shift):
        # Upstream Self-Forcing samples on the unshifted training grid, then
        # applies Wan's rational shift and clamps the resulting model time.
        level = torch.randint(0, 1000, (batch,), device=device).float() / 1000
        sigma = shift * level / (1 + (shift - 1) * level)
        sigma = sigma.clamp(0.02, 0.98)
        return sigma * 1000, sigma[:, None, None, None, None]

    def generator_loss(self, generated, real_model, fake_model,
                       real_backend, fake_backend,
                       real_context, real_negative_context,
                       fake_context, fake_negative_context,
                       rows, args, device):
        frames = generated.shape[1]
        timestep, sigma = self.score_timestep(
            generated.shape[0], device, args.score_timestep_shift,
        )
        with torch.no_grad():
            noise = torch.randn_like(generated)
            noisy = (1 - sigma) * generated.detach() + sigma * noise
            fake_flow = _score_flow(
                fake_backend, fake_model, noisy, timestep, fake_context, rows, device,
            )
            if args.fake_guidance_scale:
                fake_uncond = _score_flow(
                    fake_backend, fake_model, noisy, timestep, fake_negative_context,
                    rows, device,
                )
                fake_flow = fake_flow + args.fake_guidance_scale * (fake_flow - fake_uncond)
            real_flow = _score_flow(
                real_backend, real_model, noisy, timestep, real_context, rows, device,
            )
            if args.real_guidance_scale:
                real_uncond = _score_flow(
                    real_backend, real_model, noisy, timestep, real_negative_context,
                    rows, device,
                )
                real_flow = real_flow + args.real_guidance_scale * (real_flow - real_uncond)
            fake_x0 = noisy - sigma * fake_flow
            real_x0 = noisy - sigma * real_flow
            gradient, self.gradient_metrics = _kl_gradient(fake_x0, real_x0, generated.detach())
        active = frames if args.gradient_frames is None else min(args.gradient_frames, frames)
        return _dmd_loss(generated, gradient, active)

    def critic_loss(self, generated, fake_model, backend, context,
                    rows, args, device):
        timestep, sigma = self.score_timestep(
            generated.shape[0], device, args.score_timestep_shift,
        )
        noise = torch.randn_like(generated)
        noisy = (1 - sigma) * generated.detach() + sigma * noise
        flow = _score_flow(
            backend, fake_model, noisy, timestep, context, rows, device,
        )
        target = noise.float() - generated.detach().float()
        return (flow.float() - target).square().mean()

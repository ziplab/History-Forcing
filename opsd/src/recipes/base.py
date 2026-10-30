"""Shared recipe contracts and training argument definitions."""

import argparse
import math
from pathlib import Path

import torch


def add_causal_cache_arguments(parser):
    from wan.kv_cache.causal import resolve_sink_frames

    parser.add_argument('--sink-frames', type=resolve_sink_frames, default=None,
                        help='Temporal KV sink frames; null uses the native sink, 0 disables it')


def add_common_training_arguments(
    parser, project_name, *, text_encoder=True, num_samples=True,
):
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=400)
    if num_samples:
        parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Samples per GPU per microbatch")
    parser.add_argument("--gradient-accumulation", type=int, default=2,
                        help="Microbatches per optimizer step; effective batch = batch-size x world-size x accumulation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--max-checkpoints", type=int, default=0,
        help="Retain at most this many completed checkpoints; 0 keeps all",
    )
    parser.add_argument("--evaluator", default=None,
                        help="Checkpoint evaluator import path, package.module:Class")
    parser.add_argument("--eval-config", type=Path, default=None,
                        help="Override the model YAML supplying evaluation arguments; defaults to --config")
    parser.add_argument(
        "--report-to", choices=("wandb", "swanlab", "none"), default="wandb"
    )
    parser.add_argument(
        "--tracker-mode", choices=("online", "offline"), default="online"
    )
    parser.add_argument("--project-name", default=project_name)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("resume", "weights"),
        default="resume",
        help="Resume trainer state when available, or load model weights only",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=0,
        help="LoRA rank; use 0 for full-parameter fine-tuning",
    )
    parser.add_argument(
        "--freeze-camera", action=argparse.BooleanOptionalAction, default=False,
        help="Freeze camera/action modules and exclude them from training LoRA",
    )
    parser.add_argument(
        "--lora-checkpoint", type=Path, default=None,
        help="Optional initial adapter_model.pt for LoRA training",
    )
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="SFT LoRA dropout probability; native OPSD/DMD rollouts require 0")
    parser.add_argument(
        "--lora-target-modules", default="all-linear",
        help=(
            "Comma-separated module names; all-linear matches OPSD-V by "
            "targeting every Linear inside each causal Wan block"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    if text_encoder:
        parser.add_argument(
            "--precompute-text-embeddings",
            action=argparse.BooleanOptionalAction,
            default=True,
        )


def validate_common_training_arguments(
    parser, args, *, num_samples=True,
):
    positive = ["max_train_steps", "log_every", "save_every", "batch_size", "gradient_accumulation"]
    if num_samples:
        positive.append("num_samples")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_checkpoints < 0:
        parser.error("--max-checkpoints cannot be negative")
    if args.evaluator is not None and ":" not in args.evaluator:
        parser.error("--evaluator must use package.module:Class")
    if args.eval_config is not None:
        if args.evaluator is None:
            parser.error("--eval-config requires --evaluator")
        if not args.eval_config.is_file():
            parser.error(f"Evaluator config does not exist: {args.eval_config}")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.lora_rank < 0:
        parser.error("--lora-rank cannot be negative")
    if args.lora_checkpoint is not None:
        if args.lora_rank == 0:
            parser.error("--lora-checkpoint requires --lora-rank")
        if args.lora_checkpoint.name != "adapter_model.pt":
            parser.error("--lora-checkpoint must be named adapter_model.pt")
        if not args.lora_checkpoint.is_file():
            parser.error(f"LoRA checkpoint does not exist: {args.lora_checkpoint}")
    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    targets = tuple(
        name.strip() for name in args.lora_target_modules.split(",")
        if name.strip()
    )
    if args.lora_rank and not targets:
        parser.error("--lora-target-modules cannot be empty when LoRA is enabled")
    args.lora_target_modules = targets
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        parser.error("Adam betas must be in [0, 1)")
    if not math.isfinite(args.adam_epsilon) or args.adam_epsilon <= 0:
        parser.error("--adam-epsilon must be finite and positive")
    if (not all(math.isfinite(value) for value in (args.weight_decay, args.max_grad_norm))
            or args.weight_decay < 0 or args.max_grad_norm <= 0):
        parser.error("weight decay must be finite and non-negative and grad norm finite and positive")
    if args.lr_warmup_steps < 0:
        parser.error("--lr-warmup-steps cannot be negative")


def per_sample_mse(output):
    difference = output.prediction.float() - output.target.float()
    if output.weight is not None:
        # Mask before squaring so ignored NaN/Inf slots have zero gradients.
        difference = torch.where(output.weight != 0, difference, 0.0)
    difference = difference.square()
    if output.weight is not None:
        difference = difference * output.weight
    total = difference.flatten(1).sum(1)
    if output.normalizer is None:
        normalizer = difference[0].numel()
    else:
        normalizer = output.normalizer
    return total / normalizer

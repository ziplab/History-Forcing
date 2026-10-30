#!/usr/bin/env python3
"""Native and autoregressive video sampling, output helpers, and the inference CLI."""

from utils import (
    _execution_device,
    inference_rng,
)

import argparse
import gc
import time
import traceback
from contextlib import ExitStack, nullcontext
from copy import copy
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm
from scheduler import observe_sampling

DEFAULT_INFERENCE_SEED = 42
DEFAULT_OUTPUT_FPS = 16


def parse_args(arguments=None):
    from backbones import add_model_arg, load_backend, selected_model
    from config import add_sample_arguments, parse_config_args

    backend = load_backend(selected_model("inference", arguments))
    parser = argparse.ArgumentParser(
        description="Inference for a video model"
    )
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Pretrained model directory")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Backend checkpoint (project model.pt or native checkpoint)")
    parser.add_argument("--data-path", type=Path,
                        required=True,
                        help="Shared latent LMDB directory")
    parser.add_argument("--parquet", type=Path,
                        help="Verify this source matches the packed LMDB before loading models")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for output MP4 videos")
    add_sample_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=1,
                        help=("Maximum same-prompt generations per GPU batch "
                              "(default: 1)"))
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_INFERENCE_SEED,
        help=f"Random seed (default: {DEFAULT_INFERENCE_SEED}; model may override)",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_OUTPUT_FPS,
        help=f"Output video FPS (default: {DEFAULT_OUTPUT_FPS}; model may override)",
    )
    parser.add_argument("--save-latent", action=argparse.BooleanOptionalAction,
                        help="Also save denoised latent in the shared inference path")
    add_model_arg(parser)
    backend.add_inference_arguments(parser)
    args = parse_config_args(parser, "inference", arguments=arguments)
    positive = {
        "batch_size": args.batch_size,
        "samples_per_prompt": args.samples_per_prompt,
        "fps": args.fps,
    }
    if args.num_samples is not None:
        positive["num_samples"] = args.num_samples
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sample_index < 0:
        parser.error("--sample-index cannot be negative")
    return args


def write_mp4(path, frames, fps):
    """Write uint8 RGB frames [T,H,W,C] with torchvision/PyAV."""
    from torchvision.io import write_video

    frames = torch.as_tensor(
        frames, dtype=torch.uint8, device="cpu"
    ).contiguous()
    write_video(
        str(path), frames, fps=fps, video_codec="libx264",
        options={"crf": "18"},
    )


def video_frames(video):
    return (
        video.permute(0, 2, 3, 1)
        .mul(255)
        .to(torch.uint8)
        .contiguous()
        .cpu()
    )


def write_condition_image(path, image):
    """Write one model-visible RGB condition, or remove a stale image."""
    from torchvision.io import write_png

    path = Path(path)
    if image is None:
        path.unlink(missing_ok=True)
        return
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Condition image must have shape [3,H,W]")
    pixels = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(dtype=torch.uint8, device="cpu")
        .contiguous()
    )
    write_png(pixels, str(path))


def shard_indices(start, count, rank, world_size):
    """Assign dataset samples to ranks without padding or duplication."""
    return range(start + rank, start + count, world_size)


class InferenceResources:
    """Own conditioning components for one inference phase, released on exit."""

    def __init__(self):
        self.text_encoder = self.codec = None
        self.identity = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exception, tb):
        if exception is not None:
            # Failed loaders/samplers retain modules in their traceback locals.
            # The frames are unwound here; keep the traceback but release locals
            # before clearing our last owning references and the CUDA allocator.
            traceback.clear_frames(tb)
        self.close()

    def load(self, backend, model_dir, device, options, rank=0):
        device = _execution_device(device)
        identity = (type(backend), Path(model_dir).resolve(), device)
        if self.identity is not None and self.identity != identity:
            raise ValueError("Inference resources belong to a different backend, directory or device")
        self.identity = identity
        if self.text_encoder is None and backend.uses_text_encoder(training=False, config=options):
            if rank == 0:
                print("Loading text encoder ...", flush=True)
            self.text_encoder = backend.text_encoder(model_dir, device)
        if self.codec is None:
            if rank == 0:
                print("Loading model codec ...", flush=True)
            self.codec = backend.codec(model_dir, device)
        return self.text_encoder, self.codec

    def close(self):
        self.text_encoder = self.codec = self.identity = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def inference_batch(backend, model, *args, **options):
    """Keep the FSDP root forward active around native sampling methods.

    Native samplers call their own condition/cache methods. Running that call
    inside the root forward materializes its parameters and retains nested
    FSDP forwards; calling the unwrapped sampler directly would bypass this.
    The temporary root dispatch is restored even when sampling fails.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel

    if not isinstance(model, FullyShardedDataParallel):
        return backend.inference(model, *args, **options)
    native = model.module
    had_forward = "forward" in native.__dict__
    previous = native.__dict__.get("forward")

    def restore():
        if had_forward:
            native.forward = previous
        else:
            native.__dict__.pop("forward", None)

    def dispatch():
        # Samplers such as ABot also call the native wrapper's ordinary forward.
        restore()
        return backend.inference(native, *args, **options)

    native.forward = dispatch
    try:
        return model()
    finally:
        restore()


@torch.no_grad()
def run_inference(args, *, backend=None, generator=None, rank=0, world_size=1,
                  device=None, process_group=None, resources=None):
    """Run isolated inference; preserve live weights, modes and training RNG.

    A supplied generator is never reloaded. Supplied resources belong to the
    caller's ``with resources`` scope; otherwise this call owns their lifetime.
    """
    modes = [] if generator is None else [(module, module.training) for module in generator.modules()]
    def phase(context):
        if world_size > 1 and dist.is_initialized():
            from engine.runtime import distributed_phase
            return distributed_phase(process_group, context)
        return nullcontext()

    device = _execution_device('cuda' if device is None else device)
    with inference_rng(args.seed, device=device), ExitStack() as stack:
        if resources is None:
            resources = stack.enter_context(InferenceResources())
        try:
            with phase("Inference output directory creation"):
                if rank == 0:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
            with phase("Inference"):
                count = _run_inference(
                    args, backend=backend, generator=generator, rank=rank,
                    world_size=world_size, device=device,
                    resources=resources, phase=phase,
                )
            if rank == 0:
                print(f"Done. {count} videos saved to {args.output_dir}")
        finally:
            for module, training in modes:
                module.training = training


def resolve_inference_sampling(args, backend):
    """Resolve explicit CLI/YAML values before checkpoint and native defaults.

    Fields resolve independently. DMD retains its generator flow grid and
    conditional prediction instead of inheriting a backbone's native CFG solver.
    Caller-owned arguments and backend defaults remain unchanged.
    """
    args = copy(args)
    training = {}
    saved_model = None
    checkpoint = Path(backend.checkpoint_file(args.checkpoint))
    if checkpoint.name == "model.pt":
        from engine.runtime import CHECKPOINT_CONFIG_NAME, load_checkpoint_config

        if checkpoint.with_name(CHECKPOINT_CONFIG_NAME).is_file():
            saved = load_checkpoint_config(checkpoint)
            training = saved["training"]
            saved_model = saved["model"].get("backend")
    dmd_record = training.get("recipe", training.get("trainer")) == "dmd"
    explicit_recipe = getattr(args, "sampling_recipe", None)
    dmd_capable = False
    if dmd_record or explicit_recipe == "dmd":
        from backbones.base import DMDGenerator
        dmd_capable = isinstance(backend, DMDGenerator)
    trained_dmd = (dmd_record and dmd_capable
                   and training.get("generator") == saved_model and saved_model is not None)
    if dmd_record and not trained_dmd:
        # Generator and fake-score exports share training identity. Its rollout
        # grid, mode and shift belong only to the named generator backend.
        training = {}
    recipe = explicit_recipe or ("dmd" if trained_dmd else "native")
    if recipe not in ("native", "dmd"):
        raise ValueError("sampling_recipe must be native or dmd")
    if recipe == "dmd" and not dmd_capable:
        raise ValueError(f"{type(backend).__name__} does not support the DMD generator sampler")
    args.sampling_recipe = recipe
    sources = {"recipe": "override" if explicit_recipe is not None else
               "checkpoint" if trained_dmd else "native"}
    step_name = "timestep_indices" if hasattr(args, "timestep_indices") else "sampling_steps"
    saved_steps = (None if trained_dmd and recipe == "native" else
                   training.get("sampling_steps", training.get("steps") if trained_dmd else None))
    # OPSD's optional text CFG controls distillation, not native inference.
    text_settings = {} if training.get("recipe", training.get("trainer")) == "opsd" else training
    fields = {
        "sampling_mode": (training.get("sampling_mode", "sde" if trained_dmd else None),
                          "sde" if recipe == "dmd" else getattr(backend, "sampling_mode", "ode")),
        step_name: (saved_steps,
                    None if recipe == "dmd" else getattr(backend, "sampling_steps", None)),
        "guidance_scale": (training.get("generator_guidance_scale", 1.0 if trained_dmd else None)
                           if recipe == "dmd" else text_settings.get("guidance_scale"),
                           1.0 if recipe == "dmd" else getattr(backend, "guidance_scale", None)),
        "negative_prompt": (text_settings.get("negative_prompt"), getattr(backend, "negative_prompt", None)),
    }
    for name, (saved, native) in fields.items():
        explicit = getattr(args, name, None)
        value = explicit if explicit is not None else saved if saved is not None else native
        sources[name] = "override" if explicit is not None else "checkpoint" if saved is not None else "native"
        if value is not None and (explicit is not None or saved is not None or recipe == "dmd"):
            setattr(args, name, value)
    if recipe == "dmd":
        from scheduler import parse_sampling_steps
        steps = getattr(args, "sampling_steps", None)
        if steps is None:
            raise ValueError("DMD inference needs checkpoint training.steps or explicit --sampling-steps")
        args.sampling_steps = parse_sampling_steps(steps)
        backend = copy(backend)
        backend.sampling_recipe = recipe
    shift = getattr(args, "timestep_shift", None)
    if shift is None:
        shift = training.get("timestep_shift")
    sources["timestep_shift"] = ("override" if getattr(args, "timestep_shift", None) is not None else
                                 "checkpoint" if shift is not None else "native")
    if shift is not None:
        if hasattr(args, "timestep_shift"):
            args.timestep_shift = float(shift)
        if hasattr(backend, "timestep_shift"):
            backend = copy(backend)
            backend.timestep_shift = float(shift)
    args._sampling_sources = sources
    return args, backend


def sampling_manifest(args, options, schedules, *, dtype=torch.bfloat16, seed=None):
    """Describe observed native updates separately from resolved input settings."""
    scale = options.get("guidance_scale", options.get("guide_scale", 1.0))
    if isinstance(scale, (tuple, list)):
        scale = list(scale)
        if scale and all(value == scale[0] for value in scale):
            scale = scale[0]
    conditional_only = scale == 1.0
    effective = {
        "schedules": schedules,
        "guidance": {"scale": scale, "formula": "positive" if conditional_only else
                     "negative" if scale == 0.0 else "negative + scale * (positive - negative)",
                     "negative_prompt": None if conditional_only else options.get("negative_prompt")},
        "noise_scale": options.get("noise_scale", 1.0),
        "noise_policy": "native",
    }
    for name in ("color_correction_strength", "sink_frames"):
        if name in options:
            effective[name] = options[name]
    return {
        "schema_version": 1,
        "status": "observed" if schedules else "unobserved",
        "seed": seed,
        "effective": effective,
        "requested_dtype": str(dtype),
        "resolved": {"recipe": getattr(args, "sampling_recipe", "native"),
                     "sources": getattr(args, "_sampling_sources", {}),
                     "settings": {name: list(value) if isinstance(value, tuple) else value
                                  for name, value in options.items()
                                  if name in {"sampling_steps", "timesteps_index", "steps", "sampling_mode",
                                              "timestep_shift", "shift", "guidance_scale", "guide_scale"}}},
    }


def _run_inference(args, *, backend, generator, rank, world_size, device,
                   resources, phase=nullcontext):
    """Load inputs and sample within the caller's RNG and resource scopes."""
    from backbones import load_backend
    from dataset import validate_dataset_path
    from config import sample_range

    with phase("Inference model setup"):
        backend = load_backend(args.model) if backend is None else backend
        args, backend = resolve_inference_sampling(args, backend)
        inference_options = backend.inference_options(args)
        if not torch.cuda.is_available():
            raise RuntimeError("Inference requires a CUDA GPU")
        validate_dataset_path(args.data_path)
        if getattr(args, 'parquet', None) is not None:
            from dataset import LatentSequenceDataset, VideoDataset, packing_identity, read_packing_source

            rows, root = read_packing_source(args.parquet, backend, args.data_path)
            source = VideoDataset(rows, root, pixel_shape=backend.pixel_shape, decoded_frames=backend.decoded_frames)
            with LatentSequenceDataset(args.data_path) as packed:
                packed.validate_packing_identity(packing_identity(source, backend, args.model_dir))
        backend.validate(
            args.model_dir,
            args.checkpoint,
            codec=True,
        )
        dtype = torch.bfloat16

        # --- load model ---
        if generator is None:
            if rank == 0:
                print("Loading model ...", flush=True)
            generator = backend.load(
                args.model_dir, args.checkpoint, device, dtype
            )
        elif rank == 0:
            print("Using the live training model for inference.", flush=True)
        generator.eval()

    from dataset import LatentSequenceDataset, dataset_metadata

    with ExitStack() as stack:
        with phase("Inference data preparation"):
            dataset = stack.enter_context(LatentSequenceDataset(args.data_path))
            data_metadata = dataset_metadata(dataset)
            sample_count = len(sample_range(len(dataset), args.sample_index, args.num_samples))
            from torch.distributed.fsdp import FullyShardedDataParallel

            replicated = isinstance(generator, FullyShardedDataParallel)
            local_indices = (range(args.sample_index, args.sample_index + sample_count)
                             if replicated else shard_indices(
                                 args.sample_index, sample_count, rank, world_size))
            if rank == 0:
                print(
                    f"Dataset: {len(dataset)} total, {sample_count} prompts, "
                    + ", ".join(
                        f"{name}={value}" for name, value in data_metadata.items()
                        if name != "samples"
                    ) + ", "
                    f"{args.samples_per_prompt} samples/prompt, "
                    f"{sample_count * args.samples_per_prompt} videos, "
                    f"{world_size} GPU(s)"
                )

        # --- load model-owned conditioning components ---
        with phase("Inference conditioning setup"):
            if local_indices:
                text_encoder, codec = resources.load(
                    backend, args.model_dir, device, inference_options, rank,
                )

            # --- main loop ---
            batches_per_prompt = (
                args.samples_per_prompt + args.batch_size - 1
            ) // args.batch_size
            progress = tqdm(
                total=len(local_indices) * batches_per_prompt,
                desc=f"[rank {rank}] Generating",
                disable=(rank != 0),
                dynamic_ncols=True,
            )
            stack.callback(progress.close)

        replica_phase = phase if replicated else nullcontext
        for prompt_index in local_indices:
            with replica_phase("Inference sample preparation"):
                sample = dataset[prompt_index]

            for sample_start in range(0, args.samples_per_prompt, args.batch_size):
                sample_end = min(
                    sample_start + args.batch_size, args.samples_per_prompt
                )
                sample_indices = list(range(sample_start, sample_end))
                stems = [
                    f"{prompt_index:04d}-{sample_index}"
                    for sample_index in sample_indices
                ]
                batch_size = len(sample_indices)

                with replica_phase("Inference generation"):
                    torch.cuda.synchronize(device)
                    batch_start = time.perf_counter()
                    seeds = [
                        (
                            args.seed + prompt_index + sample_index * len(dataset)
                        )
                        for sample_index in sample_indices
                    ]
                    with inference_rng(seeds[0], device=device), observe_sampling() as schedules:
                        videos, latents, condition_images = inference_batch(
                            backend,
                            generator,
                            [sample] * batch_size,
                            text_encoder,
                            codec,
                            seeds,
                            device,
                            dtype=dtype,
                            **inference_options,
                        )

                with replica_phase("Inference output writing"):
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - batch_start
                    if sample_start == 0 and (not replicated or rank == 0):
                        condition_image = (
                            None
                            if condition_images is None
                            else condition_images[0]
                        )
                        write_condition_image(
                            args.output_dir / f"{prompt_index:04d}_condition.png",
                            condition_image,
                        )
                    for batch_index, stem in enumerate(stems):
                        if replicated and rank != 0:
                            continue
                        sampling_path = args.output_dir / f"{stem}_sampling.json"
                        latent_path = args.output_dir / f"{stem}_latent.pt"
                        # An overwrite must not pair the new video with old
                        # sampling metadata or a stale fallback latent.
                        for path in (sampling_path, latent_path):
                            path.unlink(missing_ok=True)
                            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
                        write_mp4(
                            args.output_dir / f"{stem}.mp4",
                            video_frames(videos[batch_index]),
                            args.fps,
                        )
                        from engine.runtime import save_json_state, save_torch_state
                        if args.save_latent:
                            save_torch_state(latent_path, latents[batch_index].cpu())
                        record = sampling_manifest(args, inference_options, schedules,
                                                   dtype=dtype, seed=seeds[batch_index])
                        save_json_state(sampling_path, record)
                    del videos, latents, condition_images

                    progress.update(1)
                    print(
                        f"[rank {rank}] saved prompt={prompt_index:04d} "
                        f"samples={sample_start}-{sample_end - 1} "
                        f"batch={batch_size} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

        return sample_count * args.samples_per_prompt


def main():
    from utils import init_distributed

    args = parse_args()
    rank, world_size, device = init_distributed()
    try:
        run_inference(args, rank=rank, world_size=world_size, device=device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

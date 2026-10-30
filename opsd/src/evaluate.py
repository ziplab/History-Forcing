#!/usr/bin/env python3
"""Evaluate videos through a configured task evaluator or native references."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from backbones import add_model_arg, load_backend
from dataset import LatentSequenceDataset
from config import add_sample_arguments, parse_config_args, sample_range
from utils import find_generated_files, init_distributed
from inference import InferenceResources

SUPPORTED_METRICS = {"psnr", "ssim", "lpips"}
DEFAULT_VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
)
SUPPORTED_VBENCH_DIMENSIONS = set(DEFAULT_VBENCH_DIMENSIONS)


def _evaluation_phase(world_size, context):
    if world_size > 1 and dist.is_initialized():
        from engine.runtime import distributed_phase
        return distributed_phase(None, context)
    return nullcontext()


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Evaluate generated videos against native references"
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metrics", default="psnr,ssim,lpips")
    parser.add_argument(
        "--vbench", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--vbench-dimensions", default=",".join(DEFAULT_VBENCH_DIMENSIONS)
    )
    add_sample_arguments(parser)
    parser.add_argument("--device", default="cuda")
    add_model_arg(parser)
    args = parse_config_args(parser, "evaluation", arguments=arguments)

    if args.model_dir is None or args.data_path is None:
        parser.error("native evaluation requires --model-dir and --data-path")

    metrics = list(dict.fromkeys(
        metric.strip() for metric in args.metrics.split(",") if metric.strip()
    ))
    unknown = set(metrics) - SUPPORTED_METRICS
    if not metrics or unknown:
        parser.error(
            "--metrics must contain only: " + ",".join(sorted(SUPPORTED_METRICS))
        )
    vbench_dimensions = list(dict.fromkeys(
        dimension.strip()
        for dimension in args.vbench_dimensions.split(",")
        if dimension.strip()
    ))
    unknown = set(vbench_dimensions) - SUPPORTED_VBENCH_DIMENSIONS
    if not vbench_dimensions or unknown:
        parser.error(
            "--vbench-dimensions must contain only: "
            + ",".join(DEFAULT_VBENCH_DIMENSIONS)
        )
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.sample_index is not None and args.sample_index < 0:
        parser.error("--sample-index cannot be negative")
    if args.samples_per_prompt <= 0:
        parser.error("--samples-per-prompt must be positive")
    args.metrics = metrics
    args.vbench_dimensions = vbench_dimensions
    return args


def clean_latent_sequence(sample):
    latent = sample.get("clean_latent")
    if latent is None:
        raise KeyError("Dataset sample does not contain clean latent")
    if latent.ndim != 4:
        raise ValueError(f"Unexpected ground-truth latent shape: {latent.shape}")
    return latent


def decode_latent_sequence(backend, codec, latent, condition=None, *, generated=False):
    """Decode shared GT or saved model output onto the target-only RGB timeline."""
    if latent.ndim != 4 or latent.shape[1] != backend.latent_shape[0]:
        raise ValueError(
            f"Expected latent shape [F,{backend.latent_shape[0]},H,W], "
            f"got {tuple(latent.shape)}"
        )
    if latent.shape[0] == 0:
        raise ValueError(
            "Latent sequence must contain at least one frame"
        )
    condition_batch = None
    condition_frames = 0 if condition is None else condition.shape[0]
    if condition is not None:
        if condition.ndim != 4 or tuple(
            condition.shape[1:]
        ) != tuple(latent.shape[1:]):
            raise ValueError(
                "Condition latent must have shape "
                f"[F,{','.join(map(str, latent.shape[1:]))}]"
            )
        if condition_frames:
            condition_batch = condition.to(
                device=latent.device, dtype=latent.dtype
            ).unsqueeze(0)
    if generated and backend.output_includes_condition:
        if latent.shape[0] <= condition_frames:
            raise ValueError("Generated latent does not extend beyond the condition")
        # The saved tensor already contains its native prefix. Do not prepend GT.
        condition_batch = None
    video, _ = backend.decode_output(
        codec, latent.unsqueeze(0), condition_batch
    )
    if generated:
        video = video[:, backend.output_prefix_frames(condition_frames):]
    return video[0].mul(2).sub(1)


def load_mp4_video(path, device, target_hw):
    from utils import read_video_frames

    frames = read_video_frames(path)
    if frames is None:
        raise ValueError(f"Generated video is empty: {path}")
    video = frames.to(device=device, dtype=torch.float32)
    video = video / 127.5 - 1.0
    if video.shape[2:] != torch.Size(target_hw):
        video = torch.nn.functional.interpolate(
            video,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
    return video


def compute_psnr(video_gen, video_gt):
    mse = (video_gen - video_gt).square().mean(dim=(1, 2, 3))
    return (10 * torch.log10(4.0 / (mse + 1e-10))).mean().item()


def compute_ssim(video_gen, video_gt):
    from skimage.metrics import structural_similarity

    gen = video_gen.cpu().numpy().transpose(0, 2, 3, 1)
    gt = video_gt.cpu().numpy().transpose(0, 2, 3, 1)
    values = [
        structural_similarity(
            gen[i], gt[i], data_range=2.0, channel_axis=-1, win_size=11
        )
        for i in range(len(gen))
    ]
    return float(np.mean(values))


def compute_lpips(video_gen, video_gt, model):
    values = [
        model(video_gen[i:i + 1], video_gt[i:i + 1]).detach().reshape(1)
        for i in range(len(video_gen))
    ]
    # Preserve one-frame activation memory and synchronize once per video/batch.
    return torch.cat(values).double().mean().item()


def load_lpips():
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("lpips is required by evaluation.metrics") from error
    return lpips.LPIPS(net="alex", verbose=False)


def prepare_dependencies(backend, model_dir, metrics, vbench, dimensions):
    """Run each evaluator's native model preparation before NCCL."""
    vae_path = model_dir / backend.codec_config(model_dir)["vae_checkpoint"]
    if not vae_path.is_file():
        raise FileNotFoundError(f"Missing Wan VAE checkpoint: {vae_path}")
    if "lpips" in metrics:
        load_lpips()
    if not vbench:
        print("Evaluation dependencies are ready", flush=True)
        return

    try:
        from vbench.utils import CACHE_DIR, init_submodules
    except ImportError as error:
        raise RuntimeError(
            "Install VBench with: pip install --no-deps vbench==0.1.5"
        ) from error

    cache_dir = Path(CACHE_DIR)
    for name in (
        "aesthetic_model", "amt_model", "clip_model", "dino_model",
        "pyiqa_model", "raft_model",
    ):
        (cache_dir / name).mkdir(parents=True, exist_ok=True)

    submodules = init_submodules(dimensions, local=True)
    if "aesthetic_quality" in submodules:
        from vbench.aesthetic_quality import get_aesthetic_model

        get_aesthetic_model(submodules["aesthetic_quality"][1])
    print("Evaluation dependencies are ready", flush=True)


def summarize(values):
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_vbench(
    output_dir, mp4_files, keys, prompts, dimensions, device,
    rank, world_size,
):
    with _evaluation_phase(world_size, "VBench preparation"):
        try:
            from vbench import VBench
        except ImportError as error:
            raise RuntimeError(
                "Install VBench with: pip install --no-deps vbench==0.1.5"
            ) from error

        vbench_dir = output_dir / "vbench"
        input_dir = vbench_dir / "inputs"
        name = "results"
        result_path = vbench_dir / f"{name}_eval_results.json"

        prompt_list = {
            f"{prompt_index:06d}-{sample_index}.mp4": prompts[prompt_index]
            for prompt_index, sample_index in keys
        }
        if rank == 0:
            input_dir.mkdir(parents=True, exist_ok=True)
            for path in input_dir.glob("*.mp4"):
                path.unlink()
            result_path.unlink(missing_ok=True)
            for prompt_index, sample_index in keys:
                key = (prompt_index, sample_index)
                if key not in mp4_files:
                    raise FileNotFoundError(f"VBench requires an MP4 for {key}")
                filename = f"{prompt_index:06d}-{sample_index}.mp4"
                destination = input_dir / filename
                destination.symlink_to(mp4_files[key].resolve())

    # VBench checkpoints require full torch.load deserialization.
    original_load = torch.load
    torch.load = lambda *a, **kw: original_load(
        *a, **{**kw, "weights_only": False}
    )
    try:
        with _evaluation_phase(world_size, "VBench initialization"):
            evaluator = VBench(device, None, str(vbench_dir))
            original_build = evaluator.build_full_info_json
            full_info_path = vbench_dir / f"{name}_full_info.json"

        def build_full_info(*args, **kwargs):
            with _evaluation_phase(world_size, "VBench metadata writing"):
                path = (original_build(*args, **kwargs) if rank == 0 else str(full_info_path))
            return path

        evaluator.build_full_info_json = build_full_info
        with _evaluation_phase(world_size, "VBench scoring"):
            evaluator.evaluate(
                videos_path=str(input_dir),
                name=name,
                prompt_list=prompt_list,
                dimension_list=dimensions,
                local=True,
                mode="custom_input",
            )
    finally:
        torch.load = original_load

    if rank != 0:
        return None, result_path
    if not result_path.is_file():
        raise RuntimeError(f"VBench did not create {result_path}")
    with open(result_path, encoding="utf-8") as file:
        results = json.load(file)
    scores = {}
    for dimension in dimensions:
        result = results.get(dimension)
        if not isinstance(result, list) or not result:
            raise ValueError(f"Invalid VBench result for {dimension}: {result}")
        scores[dimension] = float(result[0])
    return scores, result_path


def _evaluate_native(args, backend, device, rank, world_size, resources):
    """Return CPU metric records after dropping decoder/LPIPS working tensors."""
    with LatentSequenceDataset(args.data_path, load_latent=True) as dataset:
        generated, mp4_files = find_generated_files(args.latents_dir)
        prompt_indices = list(sample_range(len(dataset), args.sample_index, args.num_samples))
        expected_keys = [
            (prompt_index, sample_index)
            for prompt_index in prompt_indices
            for sample_index in range(args.samples_per_prompt)
        ]
        missing_keys = [key for key in expected_keys if key not in generated]
        if missing_keys:
            preview = ", ".join(f"{i}-{j}" for i, j in missing_keys[:8])
            raise FileNotFoundError(
                f"Missing {len(missing_keys)} generated samples in "
                f"{args.latents_dir}: {preview}"
            )
        if args.vbench:
            missing_mp4 = [key for key in expected_keys if key not in mp4_files]
            if missing_mp4:
                preview = ", ".join(f"{i}-{j}" for i, j in missing_mp4[:8])
                raise FileNotFoundError(
                    f"VBench is enabled but {len(missing_mp4)} MP4 files are "
                    f"missing in {args.latents_dir}: {preview}"
                )

        if args.vbench and len(expected_keys) < world_size:
            raise ValueError(
                f"VBench needs at least one video per rank, but got "
                f"{len(expected_keys)} videos for {world_size} ranks"
            )
        if args.vbench and world_size > 1 and device.type != "cuda":
            raise ValueError("Multi-process VBench evaluation requires CUDA")
        if rank == 0:
            print(
                f"Evaluating {len(expected_keys)} generated videos from "
                f"{len(prompt_indices)} prompts against {len(dataset)} LMDB "
                f"samples on {world_size} GPU(s)"
            )
        local_prompt_indices = prompt_indices[rank::world_size]
        if not local_prompt_indices:
            return {}, mp4_files, prompt_indices, expected_keys
        resources.codec = backend.codec(args.model_dir, device)

        lpips_model = None
        if "lpips" in args.metrics:
            lpips_model = load_lpips().to(device).eval()

        per_sample = {}

        for prompt_index in tqdm(
            local_prompt_indices,
            desc=f"[rank {rank}] Evaluating",
            disable=(rank != 0),
        ):
            sample = dataset[prompt_index]
            prompt = sample["prompt"]
            clean_latent = clean_latent_sequence(sample).to(device)
            condition = sample.get("condition_latent")
            if condition is not None:
                condition = condition.to(device)
            condition_frames = 0 if condition is None else condition.shape[0]
            video_gt_full = decode_latent_sequence(
                backend, resources.codec, clean_latent, condition
            )
            valid_frame_counts = {
                backend.decoded_output_frames(frames, condition_frames)
                for frames in range(1, clean_latent.shape[0] + 1)
            }
            for sample_index in range(args.samples_per_prompt):
                key = (prompt_index, sample_index)
                source, generated_path = generated[key]

                if source == "latent":
                    latent = torch.load(
                        generated_path, map_location=device, weights_only=True
                    )
                    if not isinstance(latent, torch.Tensor):
                        raise TypeError(f"Expected tensor in {generated_path}")
                    video_gen = decode_latent_sequence(
                        backend, resources.codec, latent, condition, generated=True
                    )
                    del latent
                else:
                    video_gen = load_mp4_video(
                        generated_path,
                        device,
                        target_hw=video_gt_full.shape[2:],
                    )
                    video_gen = video_gen[backend.output_prefix_frames(condition_frames):]

                if video_gen.shape[0] not in valid_frame_counts:
                    raise ValueError(
                        f"Generated video length {video_gen.shape[0]} does not "
                        f"match the target-only timeline for {key}"
                    )
                if video_gen.shape[0] > video_gt_full.shape[0]:
                    raise ValueError(
                        f"Generated video has {video_gen.shape[0]} frames, but "
                        f"ground truth only has {video_gt_full.shape[0]} for {key}"
                    )
                video_gt = video_gt_full[:video_gen.shape[0]]
                if video_gen.shape != video_gt.shape:
                    raise ValueError(
                        f"Generated video shape {tuple(video_gen.shape)} does not "
                        f"match ground truth {tuple(video_gt.shape)} for {key}"
                    )
                result = {}
                if "psnr" in args.metrics:
                    result["psnr"] = compute_psnr(video_gen, video_gt)
                if "ssim" in args.metrics:
                    result["ssim"] = compute_ssim(video_gen, video_gt)
                if "lpips" in args.metrics:
                    result["lpips"] = compute_lpips(
                        video_gen, video_gt, lpips_model
                    )

                per_sample[f"{prompt_index}-{sample_index}"] = {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "source": source,
                    "frames": len(video_gt),
                    **result,
                }
                del video_gen, video_gt
            del video_gt_full, clean_latent, condition, valid_frame_counts
    return per_sample, mp4_files, prompt_indices, expected_keys


@torch.no_grad()
def main():
    from engine.evaluation import load_evaluator
    from engine.runtime import save_json_state
    from config import load_config, stage_values

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--config', type=Path)
    selector.add_argument('--all-checkpoints', action='store_true',
                          help='Evaluate base and retained checkpoints, then compare')
    selected, arguments = selector.parse_known_args()
    if selected.config is not None:
        config_path, config = load_config(selected.config)
        train = stage_values(config, 'train') or {}
        standalone = config.get('evaluation', {}).get('evaluator')
        evaluator = load_evaluator(standalone or train.get('evaluator'),
                                  config_path if standalone else train.get('eval_config') or config_path)
        if evaluator is not None:
            try:
                if selected.all_checkpoints:
                    evaluator.run_all(config_path, arguments)
                else:
                    evaluator.run(config_path, arguments)
            finally:
                evaluator.close()
            return
    if selected.all_checkpoints:
        selector.error('--all-checkpoints requires a configured task evaluator')
    args = parse_args()
    backend = load_backend(args.model)
    preflight = os.environ.get("PREPARE_EVALUATION") == "1"
    if preflight or "RANK" not in os.environ:
        prepare_dependencies(
            backend, args.model_dir, args.metrics, args.vbench, args.vbench_dimensions
        )
    if preflight:
        return

    rank, world_size, device = init_distributed(args.device)
    try:
        output_dir = args.output_dir or args.latents_dir / "evaluation"
        with _evaluation_phase(world_size, "Evaluation output directory creation"):
            if rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)

        with InferenceResources() as resources:
            with _evaluation_phase(world_size, "Native evaluation metrics"):
                per_sample, mp4_files, prompt_indices, expected_keys = _evaluate_native(
                    args, backend, device, rank, world_size, resources,
                )

        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, per_sample)
        else:
            gathered = [per_sample]

        per_sample = {key: value for part in gathered for key, value in part.items()}
        prompts = {value["prompt_index"]: value["prompt"] for value in per_sample.values()}
        source_counts = {
            source: sum(value["source"] == source for value in per_sample.values())
            for source in ("latent", "mp4")
        }
        summary = {metric: summarize([value[metric] for value in per_sample.values()])
                   for metric in args.metrics} if per_sample else {}
        if rank == 0:
            print("Evaluation results:")
            for metric, values in summary.items():
                print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")

        vbench = None
        if args.vbench:
            with _evaluation_phase(world_size, "VBench evaluation"):
                scores, result_path = run_vbench(
                    output_dir,
                    mp4_files,
                    expected_keys,
                    prompts,
                    args.vbench_dimensions,
                    device,
                    rank,
                    world_size,
                )
                if rank == 0:
                    vbench = {
                        "dimensions": args.vbench_dimensions,
                        "scores": scores,
                        "results_file": str(result_path),
                    }
                    print("VBench results:")
                    for dimension, score in scores.items():
                        print(f"  {dimension}: {score:.4f}")

        report = {
            "data_path": str(args.data_path),
            "generated_dir": str(args.latents_dir),
            "metrics": args.metrics,
            "prompt_indices": prompt_indices,
            "num_prompts": len(prompt_indices),
            "samples_per_prompt": args.samples_per_prompt,
            "num_generated_videos": len(per_sample),
            "source_counts": source_counts,
            "summary": summary,
            "per_sample": per_sample,
        }
        with _evaluation_phase(world_size, "Evaluation report writing"):
            if rank == 0:
                if vbench is not None:
                    report["vbench"] = vbench
                report_path = output_dir / "evaluation_report.json"
                save_json_state(report_path, report)
                print(f"Report saved to {report_path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

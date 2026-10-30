"""Generate long videos with DMD."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys
import time

# Running this file directly must resolve pipeline/utils/wan from the main DMD.
DMD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DMD_ROOT))

from long_video.config import load_settings, validate_model_config, validate_seed


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prompts(prompt, prompt_file):
    values = [prompt] if prompt is not None else Path(prompt_file).read_text(encoding="utf-8-sig").splitlines()
    values = [value.strip() for value in values if value.strip()]
    if not values:
        raise ValueError("At least one non-empty prompt is required")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Our DMD checkpoint model.pt or compact EMA file")
    parser.add_argument("--config", help="Model config; default: config.yaml beside the checkpoint")
    parser.add_argument("--inference-config", default=str(DMD_ROOT.parent / "configs/long_video/inference.yaml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", help="One prompt per line; batch size remains 1")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--weights", choices=("ema", "raw"), help="Default: ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="output/demo/long_video")
    parser.add_argument("--save-latents", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from evaluation.generator import DMDGenerator, resolve_config, write_json
    from utils.paths import load_config, resource_path

    settings_path = resource_path(args.inference_config).resolve()
    settings = load_settings(settings_path)
    generation = settings["generation"]
    seed = validate_seed(generation["seed"] if args.seed is None else args.seed)
    weights = args.weights or generation["weights"]
    checkpoint = resource_path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = resolve_config(args.config, checkpoint)
    model_config = load_config(config_path)
    validate_model_config(model_config)
    prompt_file = resource_path(args.prompt_file) if args.prompt_file else None
    prompts = read_prompts(args.prompt, prompt_file)
    output = resource_path(args.output_dir).resolve()

    import numpy as np
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full video generation requires a CUDA device")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_num_threads(generation["num_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # demo_utils.memory binds its device when imported by the native pipeline.
    from evaluation.video import valid_video, video_info, write_video_atomic
    from long_video.memrope import apply_memrope

    generator = DMDGenerator(str(config_path), str(checkpoint), weights, device)
    options = apply_memrope(generator.pipeline, settings["memrope"])
    model = generator.pipeline.generator.model
    if len(model.blocks) != 30 or model.num_heads != 12 or model.dim != 1536:
        raise ValueError("Use the Wan2.1-T2V-1.3B backbone")

    identity = {
        "method": "Ours_memrope", "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint), "weights": weights,
        "model_config": str(config_path), "model_config_sha256": file_sha256(config_path),
        "generation": dict(generation, weights=weights, seed=seed), "official_options": options,
        "implementation_sha256": {
            "adapter": file_sha256(Path(__file__).with_name("memrope.py")),
            "attention": file_sha256(Path(__file__).parent / "third_party/memrope/attention.py"),
        },
        "save_latents": args.save_latents,
    }
    frames = generation["latent_frames"]
    pixel_frames = 4 * frames - 3
    output.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts):
        path = output / f"prompt_{index:04d}_seed{seed}.mp4"
        receipt_path = path.with_suffix(".json")
        sample_identity = dict(identity, prompt=prompt, prompt_index=index, seed=seed)
        if path.exists() or receipt_path.exists():
            if path.is_file() and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                latent_path = path.with_suffix(".latents.pt")
                latents_ok = not args.save_latents or (latent_path.is_file() and file_sha256(latent_path) == receipt.get("latents_sha256"))
                if (receipt.get("identity") == sample_identity and latents_ok
                        and file_sha256(path) == receipt.get("video_sha256")
                        and valid_video(path, pixel_frames, generation["fps"], generation["width"], generation["height"])):
                    print(json.dumps({"status": "reused", "video": str(path)}), flush=True)
                    continue
            raise FileExistsError(f"Existing output does not match this run: {path}; choose another --output-dir")

        # Seed each video before sampling.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        noise = torch.randn((1, frames, 16, 60, 104), device=device, dtype=torch.bfloat16)
        noise_sha = hashlib.sha256(noise.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        result = generator.pipeline.inference(noise=noise, text_prompts=[prompt],
            return_latents=args.save_latents, video_output_device="cpu")
        video, latents = result if args.save_latents else (result, None)
        if tuple(video.shape) != (1, pixel_frames, 3, generation["height"], generation["width"]):
            raise RuntimeError(f"Unexpected video shape: {tuple(video.shape)}")
        if not torch.isfinite(video).all():
            raise RuntimeError("The generated video contains non-finite values")
        elapsed = time.monotonic() - started
        write_video_atomic(path, video[0].permute(0, 2, 3, 1) * 255,
                           fps=generation["fps"], options={"crf": str(generation["crf"])})
        receipt = {"identity": sample_identity, "initial_noise_sha256": noise_sha,
            "video": str(path), "video_sha256": file_sha256(path), "video_info": video_info(path),
            "inference_and_decode_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        if latents is not None:
            latent_path = path.with_suffix(".latents.pt")
            temporary = latent_path.with_name("." + latent_path.name + ".partial")
            torch.save(latents.cpu(), temporary)
            temporary.replace(latent_path)
            receipt.update(latents=str(latent_path), latents_sha256=file_sha256(latent_path))
        write_json(receipt_path, receipt)
        print(json.dumps({"status": "completed", "video": str(path), "seed": seed}), flush=True)
        generator.pipeline.kv_cache1 = None
        generator.pipeline.crossattn_cache = None
        generator.clear_cache()
        del video, latents, result, noise


if __name__ == "__main__":
    main()

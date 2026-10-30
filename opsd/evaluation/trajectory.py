"""Shared packed-data inference and verified per-sample video registration."""

import json
import subprocess
import sys
from pathlib import Path

from evaluation.io import write_text

ROOT = Path(__file__).resolve().parents[1]


def _write(path, data):
    write_text(Path(path), json.dumps(data, indent=2, allow_nan=False) + "\n")


def _configured_packed_trajectory(args, data, expected_fps):
    sys.path.insert(0, str(ROOT / "src"))
    from omegaconf import OmegaConf

    from inference import parse_args
    from config import _path_from_repo, load_config, stage_argument

    _, config = load_config(args.inference_config)
    packed = data / 'packed' if data.is_dir() else _path_from_repo(stage_argument(config, 'inference', 'data_path'))
    if data.is_dir():
        config.paths.data_path = str(packed)
    config.inference.data_path = str(packed)
    config.inference.parquet = str(data) if data.suffix == '.parquet' else None
    config.inference.checkpoint = str(args.checkpoint.absolute())
    config.inference.output_dir = str(args.output_dir.resolve())
    config.inference.use_data_length = True
    config.inference.save_latent = False
    # Evaluation owns its task list, independently of quick inference subsets.
    config.inference.sample_index = 0
    config.inference.num_samples = 'all'
    config.inference.samples_per_prompt = 1
    config.inference.pop("num_latent_frames", None)
    resolved = parse_args(["--config", str(args.inference_config),
                           "--checkpoint", config.inference.checkpoint,
                           "--output-dir", str(args.output_dir.resolve()),
                           "--data-path", str(packed)])
    config.inference.fps, config.inference.seed = resolved.fps, resolved.seed
    options = OmegaConf.to_container(config.inference, resolve=True)
    fps, seed = int(options["fps"]), int(options["seed"])
    if fps != expected_fps:
        raise ValueError("Prepared and inference FPS differ")
    return config, options, fps, seed


def _infer_and_register_trajectory(args, config, options, inputs, protocol,
                                   inference_runner=None):
    from omegaconf import OmegaConf

    config_path = args.output_dir / "inference.yaml"
    def write_inputs():
        if getattr(args, "rank", 0) == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for item in inputs:
                (args.output_dir / item['task_id'] / 'result.json').unlink(missing_ok=True)
                (args.output_dir / f"{item['row_index']:04d}-0_sampling.json").unlink(missing_ok=True)
            _write(args.output_dir / "trajectory_inputs.json", inputs)
            OmegaConf.save(config, config_path)

    if getattr(args, "world_size", 1) > 1:
        from engine.runtime import distributed_call
        distributed_call(write_inputs, args.process_group, "Evaluation input writing")
    else:
        write_inputs()
    if inference_runner is None:
        subprocess.run([sys.executable, str(ROOT / "src/inference.py"),
                        "--config", str(config_path)], check=True, cwd=ROOT)
    else:
        inference_runner(config_path)
    if getattr(args, "rank", 0) != 0:
        return
    requested = {key: value for key, value in options.items()
                 if key not in {"checkpoint", "output_dir", "save_latent"}}
    fps = int(options["fps"])
    for item in inputs:
        index, task_id = item["row_index"], item["task_id"]
        raw = args.output_dir / f"{index:04d}-0.mp4"
        sampling_path = raw.with_name(raw.stem + "_sampling.json")
        if not sampling_path.is_file():
            raise ValueError(f"Missing observed inference sampling record: {task_id}")
        sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
        if (sampling.get("schema_version") != 1 or sampling.get("status") != "observed"
                or not sampling.get("effective", {}).get("schedules")
                or sampling.get("seed") != item["seed"]):
            raise ValueError(f"Unverified inference sampling record: {task_id}")
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=nb_frames,r_frame_rate", "-of", "json", str(raw)]))["streams"][0]
        if int(probe["nb_frames"]) != item["video_frames"] or probe["r_frame_rate"] != f"{fps}/1":
            raise ValueError(f"Wrong prepared rollout frames/FPS: {task_id}: {probe}")
        video = args.output_dir / task_id / f"{task_id}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        result_path = video.parent / "result.json"
        raw.replace(video)
        condition = args.output_dir / f"{index:04d}_condition.png"
        if condition.is_file():
            condition.replace(video.parent / "condition.png")
        _write(result_path, dict(
            status="completed", model=config.model, fps=fps,
            checkpoint=str(args.checkpoint.resolve()),
            player_protocol=protocol,
            max_video_seconds=None, sampling_manifest=sampling,
            requested_inference_options=requested,
            trajectory_inputs=item, video_path=str(video.resolve()),
        ))
        sampling_path.unlink()


def preset_metadata(source: Path) -> dict:
    """Read the benchmark contract stored beside trajectories or inside Parquet."""
    if source.suffix == ".parquet":
        import pyarrow.parquet as pq

        layout = json.loads((pq.read_schema(source).metadata or {}).get(b"world_model", b"null"))
        if not isinstance(layout, dict) or layout.get("version") != 1 or "evaluation" not in layout:
            raise ValueError("Parquet needs world_model evaluation metadata")
        return layout["evaluation"]
    return {key: json.loads((source / f"{key}.json").read_text(encoding="utf-8"))
            for key in ("native_plans", "summary") if (source / f"{key}.json").is_file()}


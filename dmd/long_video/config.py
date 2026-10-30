"""Long video inference settings."""
from collections.abc import Mapping
import math

DEFAULT_MEMROPE_OPTIONS = {
    "cache_frames": 12,
    "sink_frames": 3,
    "recent_frames": 4,
    "long_memory_frames": 1,
    "short_memory_frames": 1,
    "compression_method": "ema",
    "ema_alpha_long": 0.01,
    "ema_alpha_short": 0.1,
    "ema_adaptive": False,
    "use_block_rope": True,
}
DEFAULT_GENERATION = {
    "latent_frames": 252,
    "height": 480,
    "width": 832,
    "fps": 16,
    "batch_size": 1,
    "dtype": "bfloat16",
    "weights": "ema",
    "seed": 0,
    "num_threads": 4,
    "crf": 23,
}


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return seed


def validate_memrope_options(options=None):
    if options is None:
        options = {}
    if not isinstance(options, Mapping) or set(options) - set(DEFAULT_MEMROPE_OPTIONS):
        raise ValueError("Unknown memory option in configs/long_video/inference.yaml")
    result = {**DEFAULT_MEMROPE_OPTIONS, **options}
    for name in ("cache_frames", "sink_frames", "recent_frames", "long_memory_frames", "short_memory_frames"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if result["long_memory_frames"] != 1 or result["short_memory_frames"] != 1:
        raise ValueError("EMA memory uses one long and one short memory frame")
    expected = sum(result[k] for k in ("sink_frames", "recent_frames", "long_memory_frames", "short_memory_frames")) + 3
    if result["cache_frames"] != expected:
        raise ValueError("cache_frames must equal sink + long memory + short memory + recent + 3 new frames")
    if result["compression_method"] != "ema":
        raise ValueError("compression_method must be ema")
    for name in ("ema_alpha_long", "ema_alpha_short"):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]) or not 0 < result[name] <= 1:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    for name in ("ema_adaptive", "use_block_rope"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    return result


def load_settings(path):
    from omegaconf import OmegaConf

    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict) or set(values) - {"generation", "memrope"}:
        raise ValueError("The demo config accepts generation and memrope mappings")
    supplied = values.get("generation", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_GENERATION):
        raise ValueError("Unknown generation settings")
    generation = {**DEFAULT_GENERATION, **supplied}
    for name in ("latent_frames", "height", "width", "fps", "batch_size", "num_threads", "crf"):
        if type(generation[name]) is not int:
            raise ValueError(f"generation.{name} must be an integer")
    if generation["latent_frames"] <= 0 or generation["latent_frames"] % 3:
        raise ValueError("latent_frames must be a positive multiple of 3")
    if (generation["height"], generation["width"], generation["fps"], generation["batch_size"], generation["dtype"]) != (480, 832, 16, 1, "bfloat16"):
        raise ValueError("This demo profile uses 832x480, 16 FPS, batch 1 and bfloat16")
    if generation["weights"] not in ("ema", "raw") or generation["num_threads"] <= 0 or not 0 <= generation["crf"] <= 51:
        raise ValueError("Invalid weights, num_threads or crf setting")
    validate_seed(generation["seed"])
    return {"generation": generation, "memrope": validate_memrope_options(values.get("memrope", {}))}


def validate_model_config(config):
    """Check the DMD sampling settings."""
    expected = {
        "num_frame_per_block": 3,
        "warp_denoising_step": True,
        "independent_first_frame": False,
        "context_noise": 0,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"Set {name}={value}")
    if list(config.get("denoising_step_list", [])) != [1000, 750, 500, 250]:
        raise ValueError("Set denoising_step_list=[1000, 750, 500, 250]")
    if config.get("denoising_step_list_first_chunk") is not None:
        raise ValueError("Set denoising_step_list_first_chunk=null")
    options = config.get("model_kwargs", {})
    if options.get("timestep_shift") != 5.0:
        raise ValueError("Set model_kwargs.timestep_shift=5.0")

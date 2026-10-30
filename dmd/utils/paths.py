"""Keep code paths local to DMD and resource paths at the repository root."""

from pathlib import Path
import os
from utils.attention import validate_local_attn_size


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
MODEL_ROOT = REPO_ROOT / "models"
CONFIG_ROOT = REPO_ROOT / "configs" / "long_video"


def resource_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path):
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    resolved = config.pop("_resolved_config", False)
    if type(resolved) is not bool:
        raise ValueError("_resolved_config must be a boolean")
    if not resolved:
        config = OmegaConf.merge(
            OmegaConf.load(CONFIG_ROOT / "default_config.yaml"), config)
    if "history_guidance_min_history_frames" in config:
        raise ValueError("Use history_guidance_min_prefix_frames for the minimum generated "
                         "prefix before a supervised chunk")
    for name in ("model_kwargs", "ar_teacher_kwargs"):
        options = config.get(name, {})
        if "local_attn_size" in options:
            validate_local_attn_size(options["local_attn_size"])
    retired = sorted(key for key in config
                     if key.startswith("memory_") or key == "use_memory_dmd")
    if retired:
        raise ValueError("Retired history options: " + ", ".join(retired)
                         + "; use use_history_guidance, history_guidance_scale, "
                         "history_guidance_max_timestep, history_joint_normalization and teacher_kv")
    for key in ("model_root", "generator_ckpt", "ar_teacher_ckpt", "real_score_ckpt",
                "fake_score_ckpt", "data_path", "prompt_path", "vis_prompt_path",
                "logdir", "wandb_save_dir"):
        value = config.get(key)
        if value:
            config[key] = str(resource_path(value))
    return config


def save_config(config, path, *, checkpoint=None):
    """Save resolved settings without depending on subsequently edited templates."""
    from omegaconf import OmegaConf

    values = OmegaConf.to_container(config, resolve=True)
    # Credentials belong to the environment, not exported experiment configs.
    for key in ("wandb_key", "wandb_api_key", "hf_token", "hf_key"):
        values.pop(key, None)
    if checkpoint is not None:
        values.update(generator_ckpt=str(Path(checkpoint).resolve()),
                      checkpoint_mode="resume")
    values["_resolved_config"] = True
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            OmegaConf.save(OmegaConf.create(values), stream, resolve=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

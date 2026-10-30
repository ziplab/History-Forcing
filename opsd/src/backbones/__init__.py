"""Direct model-backend loading; no aliases or global registry."""

import importlib
import argparse

from .base import (
    InteractiveSession,
    InteractiveSpec,
    LatentInteractiveSession,
    STANDARD_PLAYER_ACTIONS,
    VideoModel,
)


DEFAULT_MODEL = "backbones.dreamx:DreamXAR"


def load_backend(path):
    parts = path.split(':') if isinstance(path, str) else ()
    if len(parts) != 2 or not all(parts):
        raise ValueError("--model must use package.module:Class")
    module_name, class_name = parts
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls()
    if not isinstance(model, VideoModel):
        raise TypeError(f"{path} must construct a VideoModel")
    return model


def add_model_arg(parser):
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Direct import path of a model backend",
    )


def selected_model(section, arguments=None, *, recipe=None):
    """Use the same CLI/stage/root precedence as the final argument parser."""
    default = "backbones.dreamx:DreamXBidirectional" if section == "train" and recipe == "flow" else DEFAULT_MODEL
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--model")
    known, _ = parser.parse_known_args(arguments)
    if known.model is not None:
        return known.model
    if known.config:
        from config import load_config, stage_argument

        _, config = load_config(known.config, recipe=recipe if section == 'train' else None)
        return stage_argument(config, section, "model", default, recipe=recipe)
    return default


__all__ = [
    "InteractiveSession", "InteractiveSpec", "LatentInteractiveSession",
    "STANDARD_PLAYER_ACTIONS", "VideoModel", "DEFAULT_MODEL",
    "add_model_arg", "load_backend",
    "selected_model",
]

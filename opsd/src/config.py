"""Shared YAML configuration and command-line defaults."""

import argparse
import os
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from recipes import DEFAULT_RECIPE, RECIPE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPO_ROOT = PROJECT_ROOT.parent

CONFIG_KEYS = {
    "paths": {
        "model_dir", "data_path", "train_manifest", "initial_checkpoint",
    },
    "launcher": {"NPROC_PER_NODE"},
    "pipeline": {"run_evaluation"},
    "environment": {"OMP_NUM_THREADS", "vbench_cache_dir", "HF_HOME", "TORCH_HOME"},
}

STAGE_SECTIONS = {"train", "inference", "evaluation"}

SHARED_ARGUMENTS = {
    "model": "model",
    "model_dir": "paths.model_dir",
    "data_path": "paths.data_path",
}

def _checkpoint_tag(step):
    return f"checkpoint-{int(step):06d}"

def _register_resolvers():
    OmegaConf.register_new_resolver(
        "checkpoint_tag", _checkpoint_tag, replace=True
    )

def _validate_config(config):
    if not OmegaConf.is_dict(config):
        raise TypeError("Config root must be a mapping")
    allowed_sections = set(CONFIG_KEYS) | STAGE_SECTIONS | {"model"}
    unknown_sections = sorted(set(config) - allowed_sections)
    if unknown_sections:
        raise ValueError(
            "Unknown config sections: " + ", ".join(unknown_sections)
        )
    for section, allowed_keys in CONFIG_KEYS.items():
        node = OmegaConf.select(config, section)
        if node is None:
            continue
        if not OmegaConf.is_dict(node):
            raise TypeError(f"Config section '{section}' must be a mapping")
        unknown = sorted(set(node) - allowed_keys)
        if unknown:
            raise ValueError(
                f"Unknown keys in config section '{section}': "
                + ", ".join(unknown)
            )
    for section in STAGE_SECTIONS:
        node = config.get(section)
        if section in config and not OmegaConf.is_dict(node):
            raise TypeError(f"Config section '{section}' must be a mapping")
    datasets = OmegaConf.select(config, 'evaluation.datasets')
    if datasets is not None and (not OmegaConf.is_dict(datasets) or not datasets or
            any(not isinstance(name, str) or not name.strip() or not OmegaConf.is_dict(options)
                for name, options in datasets.items())):
        raise TypeError('evaluation.datasets must map dataset names to evaluator options')
    for key in ('launcher.NPROC_PER_NODE', 'environment.OMP_NUM_THREADS'):
        value = OmegaConf.select(config, key)
        if value is not None:
            try:
                valid = type(value) in (int, str) and int(value) > 0
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"Config field '{key}' must be a positive integer")
    if 'train' in config:
        training_recipe(config)
        for name in RECIPE_NAMES:
            if name in config.train and not OmegaConf.is_dict(config.train[name]):
                raise TypeError(f"Config section 'train.{name}' must be a mapping")
    for section in ('', 'paths', *sorted(STAGE_SECTIONS), *(f'train.{name}' for name in RECIPE_NAMES)):
        node = OmegaConf.select(config, section) if section else config
        if not OmegaConf.is_dict(node):
            continue
        fields = CONFIG_KEYS['paths'] if section == 'paths' else ('model', 'model_dir', 'data_path')
        for name in fields:
            value = node.get(name)
            expected = str if name == 'model' else (str, os.PathLike)
            if value is not None and not isinstance(value, expected):
                field = f'{section}.{name}' if section else name
                kind = 'a string' if name == 'model' else 'a path'
                raise TypeError(f"Config field '{field}' must be {kind}")

def _load_config(path, parents=()):
    """Merge relative parents before resolving references to overridden values."""
    if not parents:
        _register_resolvers()
    path = Path(path).expanduser().resolve()
    if path in parents:
        raise ValueError("Config inheritance cycle: " + " -> ".join(map(str, (*parents, path))))
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config = OmegaConf.load(path)
    if not OmegaConf.is_dict(config):
        raise TypeError("Config root must be a mapping")
    sources = config.pop('extends', [])
    if isinstance(sources, str):
        sources = [sources]
    elif OmegaConf.is_list(sources):
        sources = OmegaConf.to_container(sources)
    if not isinstance(sources, list) or any(not isinstance(source, str) or not source.strip() for source in sources):
        raise TypeError("Config 'extends' must be a path or a list of paths")
    merged = OmegaConf.create({})
    for source in sources:
        parent_path = (path.parent / source).resolve()
        parent = _load_config(parent_path, (*parents, path))
        _redirect_config_references(parent, parent_path, path)
        merged = OmegaConf.merge(merged, parent)
    return OmegaConf.merge(merged, config)

def _redirect_config_references(config, source, target):
    """Carry literal self references with their config; keep interpolations lazy."""
    raw = OmegaConf.to_container(config, resolve=False)
    train = raw.get('train')
    nodes = [('evaluation', raw.get('evaluation')), ('train', train)]
    if isinstance(train, dict):
        nodes += [(f'train.{name}', train.get(name)) for name in RECIPE_NAMES]
    for section, node in nodes:
        if not isinstance(node, dict):
            continue
        for name in ('inference_config', 'eval_config'):
            value = node.get(name)
            if (isinstance(value, str) and value
                    and not OmegaConf.is_interpolation(OmegaConf.select(config, section), name)
                    and _path_from_repo(value).resolve() == source):
                OmegaConf.update(config, f'{section}.{name}', str(target))

def _resolve_config(config, overrides=(), *, recipe=None):
    """Apply recipe selection and field overrides before resolving references."""
    if overrides:
        if any('=' not in value for value in overrides):
            raise ValueError('Configuration overrides must use key=value')
        override = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override)
    if recipe is not None and OmegaConf.is_dict(config.get('train')):
        config.train.recipe = recipe
    _validate_config(config)
    OmegaConf.resolve(config)
    return config

def load_config(path, overrides=(), *, recipe=None):
    path = Path(path).expanduser().resolve()
    return path, _resolve_config(_load_config(path), overrides, recipe=recipe)

def evaluation_dataset_config(config, name, *, config_path=None):
    """Select named benchmark inputs while sharing model, sampling and RNG."""
    datasets = OmegaConf.select(config, 'evaluation.datasets')
    if not OmegaConf.is_dict(datasets) or name not in datasets:
        raise ValueError(f'Unknown evaluation dataset: {name}')
    selected = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    common = {key: value for key, value in selected.evaluation.items()
              if key not in {'datasets', 'evaluator'}}
    selected.evaluation = OmegaConf.merge(common, datasets[name])
    if config_path is not None:
        selected.evaluation.inference_config = str(Path(config_path).resolve())
    packed = selected.evaluation.pop('data_path')
    selected.inference.parquet = selected.evaluation.prepared_data
    selected.inference.data_path = packed
    selected.inference.output_dir = str(Path(selected.inference.output_dir) / name)
    for recipe in RECIPE_NAMES:
        if OmegaConf.select(selected, f'train.{recipe}.evaluator') is not None:
            selected.train[recipe].evaluator = selected.evaluation.evaluator
    return selected

def _run_config(config_path, config, stage):
    """Persist overrides and keep task references on the same immutable config."""
    section = 'train' if stage in {'pack', 'train', 'experiment'} else 'inference'
    output = _path_from_repo(stage_argument(config, section, 'output_dir', 'output/opsd'))
    directory = output / '.configs'
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / f'{uuid4().hex}.yaml'
    _redirect_config_references(config, config_path, snapshot)
    OmegaConf.save(config, snapshot)
    return snapshot

def _path_from_repo(value):
    """Anchor input paths without losing symlink filenames or adjacent release assets."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path

def training_recipe(config, override=None):
    selected = (override if override is not None else DEFAULT_RECIPE if config is None else
                OmegaConf.select(config, 'train.recipe', default=DEFAULT_RECIPE))
    if selected not in RECIPE_NAMES:
        raise ValueError('train.recipe must be one of: ' + ', '.join(RECIPE_NAMES))
    return selected

def stage_values(config, section, *, recipe=None):
    """Select one recipe without leaking another recipe's arguments into training."""
    node = OmegaConf.select(config, section)
    if section != 'train' or not OmegaConf.is_dict(node):
        return node
    selected = training_recipe(config, recipe)
    common = {key: value for key, value in node.items() if key not in {*RECIPE_NAMES, 'recipe'}}
    if any(name in node for name in RECIPE_NAMES):
        if selected not in node:
            raise ValueError(f"Training recipe '{selected}' is not configured")
        return OmegaConf.merge(common, node[selected])
    return OmegaConf.create(common)

def stage_argument(config, section, name, default=None, *, recipe=None):
    """Resolve stage, shared and native defaults; null adds no override."""
    node = stage_values(config, section, recipe=recipe)
    value = node[name] if OmegaConf.is_dict(node) and name in node else None
    if value is None:
        key = SHARED_ARGUMENTS.get(name)
        value = OmegaConf.select(config, key) if key is not None else None
    return default if value is None else value

def sample_count(value):
    if value is None or str(value).lower() == "all":
        return None
    return int(value)

def add_sample_arguments(parser):
    """Shared inference/evaluation selection; indices always refer to dataset rows."""
    parser.add_argument("--num-samples", type=sample_count, default=None,
                        help="Number of prompts from --sample-index (default: all remaining)")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="First dataset sample (default: 0)")
    parser.add_argument("--samples-per-prompt", type=int, default=1,
                        help="Number of videos per prompt (default: 1)")

def sample_range(size, start, count):
    """Resolve one nonempty, bounded row range for generation or evaluation."""
    if not 0 <= start < size:
        raise ValueError(f"sample_index={start} is outside dataset size {size}")
    count = size - start if count is None else count
    if count <= 0 or start + count > size:
        raise ValueError(f"Requested sample range [{start}, {start + count}) exceeds dataset size {size}")
    return range(start, start + count)

def parse_config_args(parser, section, ignored_keys=(), arguments=None, *, recipe=None):
    """Use the selected stage/recipe as argparse defaults; CLI flags win."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    known, _ = pre_parser.parse_known_args(arguments)
    parser.add_argument("--config", type=Path, default=None)

    if known.config is not None:
        try:
            config_path = known.config.expanduser().resolve()
            config = _load_config(config_path)
            train = config.get('train')
            configured_recipe = (OmegaConf.to_container(train, resolve=False).get('recipe', DEFAULT_RECIPE)
                                 if OmegaConf.is_dict(train) else None)
            config = _resolve_config(config, recipe=recipe if section == 'train' else None)
            node = stage_values(config, section, recipe=recipe)
        except (
            FileNotFoundError, OmegaConfBaseException, TypeError, ValueError,
        ) as error:
            parser.error(str(error))
        if node is None:
            parser.error(f"Config {config_path} has no '{section}' section")
        values = OmegaConf.to_container(node, resolve=True)
        if not isinstance(values, dict):
            parser.error(f"Config section '{section}' must be a mapping")

        actions = {action.dest: action for action in parser._actions}
        for name in SHARED_ARGUMENTS:
            value = stage_argument(config, section, name, recipe=recipe)
            if name in actions and values.get(name) is None and value is not None:
                values[name] = value
        ignored_keys = set(ignored_keys)
        # Null declares no override, including fields owned by another backend.
        unknown = sorted(name for name in set(values) - set(actions) - ignored_keys
                         if values[name] is not None)
        if unknown:
            parser.error(
                f"Unknown keys in config section '{section}': "
                + ", ".join(unknown)
            )

        defaults = {}
        for name, value in values.items():
            if name in ignored_keys:
                continue
            if value is None:
                continue
            action = actions[name]
            if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction,
                                   argparse._StoreFalseAction)):
                if not isinstance(value, bool):
                    parser.error(
                        f"Config value '{section}.{name}' must be boolean"
                    )
            elif action.type is not None:
                try:
                    if action.type is int and (isinstance(value, bool) or not isinstance(value, (int, str))):
                        raise ValueError('must be an integer')
                    if action.type is float and isinstance(value, bool):
                        raise ValueError('must be a number')
                    value = _path_from_repo(value) if action.type is Path else action.type(value)
                except (TypeError, ValueError) as error:
                    parser.error(
                        f"Invalid config value '{section}.{name}': {error}"
                    )
            if action.choices is not None and value not in action.choices:
                parser.error(
                    f"Config value '{section}.{name}' must be one of "
                    + ", ".join(map(str, action.choices))
                )
            defaults[name] = value
            action.required = False
        parser.set_defaults(**defaults, config=config_path)

    args = parser.parse_args(arguments)
    if known.config is not None and section == 'train' and recipe is not None and recipe != configured_recipe:
        snapshot = _run_config(config_path, config, 'train')
        if getattr(args, 'eval_config', None) == config_path:
            args.eval_config = snapshot
        args.config = snapshot
    return args

def _select(config, key, default=None):
    value = OmegaConf.select(config, key, default=default)
    value = default if value is None else value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value

#!/usr/bin/env python3
"""Unified video training entry point."""

import argparse
from pathlib import Path

from backbones import add_model_arg, load_backend, selected_model
from engine.dmd import run_dmd_training
from engine.opsd import run_opsd_training
from engine.trainer import run_training
from config import load_config, parse_config_args, training_recipe
from recipes import RECIPE_NAMES
from recipes.dmd import DMDRecipe
from recipes.flow import FlowRecipe
from recipes.opsd import OPSDRecipe

RECIPES = {
    "flow": (FlowRecipe, run_training),
    "opsd": (OPSDRecipe, run_opsd_training),
    "dmd": (DMDRecipe, run_dmd_training),
}


def parse_args(recipe, arguments=None):
    recipe_name = next(name for name, (cls, _) in RECIPES.items() if isinstance(recipe, cls))
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    model_path = selected_model("train", arguments, recipe=recipe_name)
    parser.set_defaults(model=model_path)
    backend = load_backend(model_path)
    recipe.add_arguments(parser, backend)
    args = parse_config_args(parser, "train", arguments=arguments, recipe=recipe_name)
    recipe.validate_arguments(parser, args)
    backend.validate_training_arguments(parser, args)
    args.checkpoint = backend.checkpoint_file(args.checkpoint)
    return args, backend


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--recipe", choices=RECIPE_NAMES
    )
    selector.add_argument('--config', type=Path)
    selected, arguments = selector.parse_known_args()
    config = load_config(selected.config, recipe=selected.recipe)[1] if selected.config is not None else None
    name = training_recipe(config, selected.recipe)
    if selected.config is not None:
        arguments += ['--config', str(selected.config)]
    recipe_type, run = RECIPES[name]
    recipe = recipe_type()
    args, backend = parse_args(recipe, arguments)
    run(args, recipe)


if __name__ == "__main__":
    main()

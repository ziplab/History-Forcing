"""Run official VBench with the shared environment's Transformers and timm APIs."""
import argparse
from functools import wraps
import importlib
from pathlib import Path
import runpy
import sys


def prepare_runtime(dimensions):
    if "motion_smoothness" in dimensions:
        import torch
        import typing
        # The released AMT checkpoint serializes its state with this stdlib alias.
        torch.serialization.add_safe_globals([(typing.OrderedDict, "typing.OrderedDict")])

    if "scene" in dimensions:
        import transformers.modeling_utils as modeling
        from transformers import pytorch_utils
        from transformers.generation import GenerationMixin

        # Tag2Text imports these helpers from their former Transformers location.
        for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices",
                     "prune_linear_layer"):
            if not hasattr(modeling, name):
                setattr(modeling, name, getattr(pytorch_utils, name))
        med = importlib.import_module("vbench.third_party.tag2Text.med")
        if not issubclass(med.BertLMHeadModel, GenerationMixin):
            class BertLMHeadModel(med.BertLMHeadModel, GenerationMixin):
                @classmethod
                def _supports_default_dynamic_cache(cls):
                    # Keep Tag2Text's original tuple-based cache interface.
                    return False
            med.BertLMHeadModel = BertLMHeadModel

    if "human_action" in dimensions:
        import timm
        module = importlib.import_module("vbench.third_party.umt.models.modeling_finetune")
        original = module.vit_large_patch16_224

        @wraps(original)
        def create_model(*args, **kwargs):
            # Recent timm forwards a cache argument to this local constructor.
            kwargs.pop("cache_dir", None)
            return original(*args, **kwargs)
        timm.models.register_model(create_model)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--dimension", nargs="+", required=True)
    args, remaining = parser.parse_known_args()
    root = args.vbench_root.resolve()
    sys.path.insert(0, str(root))
    prepare_runtime(args.dimension)
    sys.argv = [str(root / "evaluate.py"), "--dimension", *args.dimension, *remaining]
    runpy.run_path(str(root / "evaluate.py"), run_name="__main__")


if __name__ == "__main__":
    main()

"""Action2V model, adapters, and checkpoint utilities for Wan."""

import math
import re
import json
from pathlib import Path
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn

VAE_TEMPORAL_STRIDE = 4


def normalize_checkpoint_keys(state_dict):
    """Strip common DDP, FSDP, and checkpoint-wrapper prefixes."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_fsdp_wrapped_module."):
            key = key[len("_fsdp_wrapped_module."):]
        key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        key = key.replace("model._checkpoint_wrapped_module.", "model.", 1)
        key = key.replace("model._orig_mod.", "model.", 1)
        if key in normalized:
            raise ValueError(f"Duplicate normalized checkpoint key: {key}")
        normalized[key] = value
    return normalized


def _lora_exclusion_pattern(module_names):
    """Match whole excluded subtrees, including nested adapter targets."""
    if not module_names:
        return None
    return "(?:" + "|".join(re.escape(name) for name in module_names) + r")(?:\..*)?"


@contextmanager
def _adapter_initialization(seed):
    """Seed adapter initialization without advancing CPU or local CUDA RNG."""
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        yield


def apply_lora(model, rank, alpha, dropout, target_modules, seed, *,
               exclude_modules=()):
    """Inject PEFT LoRA into Wan and freeze its base weights."""
    if rank == 0:
        return model
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig, inject_adapter_in_model

    with _adapter_initialization(seed):
        # Every distributed rank must start from the same adapter weights.
        # fork_rng keeps the rank-specific training RNG unchanged afterwards.
        model.model = inject_adapter_in_model(
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=_resolve_lora_targets(model, target_modules),
                exclude_modules=_lora_exclusion_pattern(exclude_modules),
                bias="none",
                init_lora_weights=True,
                inference_mode=False,
            ),
            model.model,
        )
    trainable_backbone = [
        name for name, parameter in model.model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_backbone or any(
        "lora_" not in name for name in trainable_backbone
    ):
        raise RuntimeError(
            "LoRA injection must freeze every non-LoRA Wan backbone parameter"
        )
    cast_training_params([model], dtype=torch.float32)
    return model


def _resolve_lora_targets(model, target_modules):
    """Resolve OPSD-V's block-local all-Linear adapter policy."""
    target_modules = tuple(target_modules)
    if target_modules != ("all-linear",):
        return list(target_modules)

    targets = set()
    for index, block in enumerate(model.model.blocks):
        name = f"blocks.{index}"
        for module_name, module in block.named_modules(prefix=name):
            if isinstance(module, nn.Linear):
                targets.add(module_name)
    if not targets:
        raise RuntimeError("Wan contains no transformer-block Linear modules")
    return sorted(targets)


def _read_lora_weights(path):
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file

        payload = load_file(str(path))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(payload, dict) and "generator_lora" in payload:
        payload = payload["generator_lora"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("LoRA checkpoint must contain a non-empty state dict")
    return payload


def load_lora_checkpoint(model, path, adapter_name="default"):
    """Load a project adapter or an upstream PEFT LoRA checkpoint."""

    load_lora_state_dict(
        model,
        _read_lora_weights(path),
        adapter_name,
    )


def _lora_key_signature(name):
    """Canonicalize wrapper and adapter names without rewriting PEFT tensors."""
    name = next(iter(normalize_checkpoint_keys({name: None})))
    for prefix in ("base_model.model.", "base_model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    match = re.fullmatch(
        r"(.*\.lora_(?:A|B))(?:\.[^.]+)?\.weight", name
    )
    return None if match is None else match.group(1) + ".weight"


def lora_state_dict(model, state_dict=None, adapter_name="default"):
    """Export one adapter with PEFT's portable state-dict convention.

    ``inject_adapter_in_model`` attaches PEFT to ``model.model`` rather than
    wrapping the outer Action2V module.  Therefore the gathered outer state is
    stripped by exactly one ``model.`` prefix before passing it to PEFT.
    """
    from peft import get_peft_model_state_dict

    outer_state = (
        model.state_dict()
        if state_dict is None
        else normalize_checkpoint_keys(state_dict)
    )
    inner_state = {
        name[len("model."):]: value
        for name, value in outer_state.items()
        if name.startswith("model.")
    }
    adapter = get_peft_model_state_dict(
        model.model,
        state_dict=inner_state,
        adapter_name=adapter_name,
        save_embedding_layers=False,
    )
    if not adapter:
        raise RuntimeError(f"Model has no {adapter_name!r} LoRA parameters")

    return adapter


def load_lora_state_dict(model, state_dict, adapter_name="default"):
    """Validate in the adapter dtype before loading portable or raw PEFT weights."""
    from peft import set_peft_model_state_dict

    state = normalize_checkpoint_keys(state_dict)
    expected = lora_state_dict(model, adapter_name=adapter_name)
    expected_by_signature = {
        _lora_key_signature(name): name for name in expected
    }
    if None in expected_by_signature or len(expected_by_signature) != len(expected):
        raise RuntimeError("PEFT exported ambiguous LoRA parameter names")

    portable = {}
    unexpected = []
    mismatched = []
    for name, value in state.items():
        signature = _lora_key_signature(name)
        target = expected_by_signature.get(signature)
        if target is None:
            unexpected.append(name)
            continue
        if target in portable:
            raise ValueError(
                f"LoRA checkpoint maps multiple tensors to {target}"
            )
        if not torch.is_tensor(value) or value.is_complex():
            raise TypeError(f"LoRA parameter {name} must be a real tensor")
        if expected[target].shape != value.shape:
            mismatched.append((name, target))
            continue
        portable[target] = value.detach().to(dtype=expected[target].dtype)
    if unexpected:
        raise ValueError(
            "Unexpected LoRA parameters: "
            f"{unexpected[:16]}"
        )
    if mismatched:
        raise ValueError(
            f"LoRA checkpoint has {len(mismatched)} shape mismatches; "
            f"first: {mismatched[:8]}"
        )
    missing = set(expected).difference(portable)
    if missing:
        raise ValueError(
            f"LoRA checkpoint is missing {len(missing)} adapter tensors; "
            f"first: {sorted(missing)[:8]}"
        )
    invalid = [
        name for name, value in portable.items()
        if not torch.isfinite(value).all().item()
    ]
    if invalid:
        raise ValueError(
            f"LoRA checkpoint contains non-finite tensors in the adapter dtype: {invalid[:8]}"
        )

    incompatible = set_peft_model_state_dict(
        model.model, portable, adapter_name=adapter_name
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "PEFT rejected canonical LoRA parameters: "
            f"{incompatible.unexpected_keys[:8]}"
        )


def merge_inference_lora(model, spec):
    """Attach, strictly load, and merge a release LoRA after base-weight loading.

    The backend chooses the transformer targets; the shared loader merges
    release adapters in FP32 before converting to the model's compute dtype.
    """
    from peft import LoraConfig, get_peft_model

    checkpoint = Path(spec.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if spec.rank <= 0 or spec.alpha <= 0 or not spec.targets:
        raise ValueError("Inference LoRA requires positive rank/alpha and target modules")
    state = _read_lora_weights(checkpoint)
    config = LoraConfig(
        r=spec.rank, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
        init_lora_weights=True, target_modules=list(spec.targets),
    )
    with _adapter_initialization(spec.seed):
        adapter = get_peft_model(model.model, config, adapter_name=spec.adapter_name)

    model.model = adapter
    load_lora_state_dict(model, state, spec.adapter_name)

    model.model = adapter.merge_and_unload().eval().requires_grad_(False)
    return model


def apply_opsd_lora(
    model, rank, alpha, dropout, target_modules, seed,
    student_adapter="student", teacher_adapter="teacher",
    *, exclude_modules=(),
):
    """Install trainable student and frozen EMA-teacher PEFT adapters."""
    if rank <= 0:
        raise ValueError("OPSD requires a positive LoRA rank")
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig, inject_adapter_in_model
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_resolve_lora_targets(model, target_modules),
        exclude_modules=_lora_exclusion_pattern(exclude_modules),
        bias="none",
        init_lora_weights=True,
        inference_mode=False,
    )
    with _adapter_initialization(seed):
        model.model = inject_adapter_in_model(
            config, model.model, adapter_name=student_adapter
        )
        model.model = inject_adapter_in_model(
            config, model.model, adapter_name=teacher_adapter
        )
    student = dict(model.model.named_parameters())
    with torch.no_grad():
        for name, target in student.items():
            marker = f".{teacher_adapter}."
            if marker not in name:
                continue
            source_name = name.replace(marker, f".{student_adapter}.")
            source = student.get(source_name)
            if source is None:
                raise RuntimeError(f"Missing student adapter tensor: {source_name}")
            target.copy_(source)
    for name, parameter in model.named_parameters():
        student_lora = f".{student_adapter}." in name
        parameter.requires_grad_(student_lora)
    set_lora_adapter(model, student_adapter, trainable_adapter=student_adapter)
    cast_training_params([model], dtype=torch.float32)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("OPSD student adapter has no trainable parameters")
    return model


def set_lora_adapter(model, adapter_name, trainable_adapter=None):
    """Select one adapter through PEFT while preserving explicit grad policy."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    found = False
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            if not any(
                adapter_name in getattr(module, name, {})
                for name in module.adapter_layer_names
            ):
                raise KeyError(f"Adapter {adapter_name!r} is missing")
            module.enable_adapters(True)
            module.set_adapter(adapter_name)
            if module.disable_adapters:
                raise RuntimeError("PEFT left the selected adapter disabled")
            active = tuple(module.active_adapters)
            if active != (adapter_name,):
                raise RuntimeError(
                    f"PEFT activated adapters {active}, expected "
                    f"{(adapter_name,)}"
                )
            found = True
    if not found:
        raise RuntimeError("Model has no PEFT adapter layers")
    if trainable_adapter is not None:
        marker = f".{trainable_adapter}."
        for name, parameter in model.named_parameters():
            if ".lora_" in name:
                parameter.requires_grad_(marker in name)


@torch.no_grad()
def update_lora_ema(model, decay, student="student", teacher="teacher"):
    parameters = dict(model.named_parameters())
    updated = 0
    marker = f".{teacher}."
    for name, target in parameters.items():
        if marker not in name:
            continue
        source = parameters.get(name.replace(marker, f".{student}."))
        if source is None:
            raise RuntimeError(f"Missing EMA source for {name}")
        target.lerp_(source, 1.0 - decay)
        updated += 1
    if not updated:
        raise RuntimeError("EMA update found no teacher adapter tensors")


class FullModelEMA:
    """CPU FP32 EMA shadow with reversible FSDP parameter swapping.

    OPSD needs the teacher for forward passes, but keeping a second full Wan
    module resident on the GPU is unnecessarily expensive.  The shadow follows
    OPSD-V's full-parameter EMA policy: keep one FP32 CPU copy, temporarily
    install it for a no-grad teacher forward, then restore the student weights.
    """

    def __init__(self, model, decay):
        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = decay
        self.shadow = {}
        self._backup = {}
        self._teacher_active = False
        with self._summon(model, writeback=False):
            for name, parameter in self._parameters(model).items():
                self.shadow[name] = parameter.detach().to(device="cpu", dtype=torch.float32, copy=True)
        if not self.shadow:
            raise RuntimeError("Full-model EMA found no parameters")

    @staticmethod
    def _name(name):
        return name.replace("_fsdp_wrapped_module.", "").removeprefix(
            "module."
        )

    @classmethod
    def _parameters(cls, model):
        parameters = {}
        for raw_name, parameter in model.named_parameters():
            name = cls._name(raw_name)
            if name in parameters:
                raise RuntimeError(f"Duplicate EMA parameter name: {name}")
            parameters[name] = parameter
        return parameters

    @staticmethod
    def _summon(model, writeback):
        from torch.distributed.fsdp import FullyShardedDataParallel

        if isinstance(model, FullyShardedDataParallel):
            return FullyShardedDataParallel.summon_full_params(
                model, writeback=writeback, rank0_only=False
            )
        return nullcontext()

    def state_dict(self):
        return dict(self.shadow)

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("Full-model EMA state must be a non-empty mapping")
        state = {self._name(name): value for name, value in state_dict.items()}
        if len(state) != len(state_dict):
            raise RuntimeError("Duplicate EMA parameter names in checkpoint")
        if set(state) != set(self.shadow):
            missing = sorted(set(self.shadow) - set(state))
            unexpected = sorted(set(state) - set(self.shadow))
            raise RuntimeError(
                "Full-model EMA checkpoint mismatch: missing="
                f"{missing[:16]} unexpected={unexpected[:16]}"
            )
        for name, target in self.shadow.items():
            source = state[name]
            if not torch.is_tensor(source):
                raise TypeError(
                    f"Full-model EMA value for {name} is not a tensor"
                )
            if source.is_complex():
                raise TypeError(f"Full-model EMA value for {name} must be real")
            if source.shape != target.shape:
                raise ValueError(
                    f"Full-model EMA shape mismatch for {name}: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}"
                )
            if not torch.isfinite(source).all().item():
                raise ValueError(f"Full-model EMA contains non-finite {name}")
            if source.dtype == torch.float64 and source.numel() and (
                source.detach().abs().max().item() > torch.finfo(target.dtype).max
            ):
                raise ValueError(f"Full-model EMA value for {name} exceeds {target.dtype} range")
        # Validate all parameters before replacing any existing teacher state.
        for name, target in self.shadow.items():
            target.copy_(state[name].detach())

    @torch.no_grad()
    def update(self, model):
        with self._summon(model, writeback=False):
            parameters = self._parameters(model)
            if set(parameters) != set(self.shadow):
                raise RuntimeError("Student and full-model EMA parameters differ")
            for name, parameter in parameters.items():
                source = parameter.detach().float().cpu()
                if parameter.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(
                        source, alpha=1.0 - self.decay,
                    )
                else:
                    self.shadow[name].copy_(source)

    @contextmanager
    def average_parameters(self, model):
        """Run a forward with EMA weights and always restore the student."""
        if self._teacher_active:
            raise RuntimeError("Full-model EMA parameter swap cannot be nested")
        self._teacher_active = True
        swapped = False
        try:
            with torch.no_grad(), self._summon(model, writeback=True):
                parameters = self._parameters(model)
                if set(parameters) != set(self.shadow):
                    raise RuntimeError(
                        "Student and full-model EMA parameters differ"
                    )
                for name, parameter in parameters.items():
                    backup = self._backup.get(name)
                    if backup is None:
                        backup = torch.empty_like(parameter, device="cpu")
                        self._backup[name] = backup
                    backup.copy_(parameter.detach())
                swapped = True
                for name, parameter in parameters.items():
                    parameter.copy_(self.shadow[name].to(
                        device=parameter.device, dtype=parameter.dtype
                    ))
            yield
        finally:
            try:
                if swapped:
                    with torch.no_grad(), self._summon(model, writeback=True):
                        parameters = self._parameters(model)
                        for name, parameter in parameters.items():
                            backup = self._backup[name]
                            parameter.copy_(backup.to(
                                device=parameter.device,
                                dtype=parameter.dtype,
                            ))
            finally:
                self._teacher_active = False


def merge_lora_state_dict(
    state_dict, rank, alpha, device, adapter_name="default",
    discard_adapters=(),
):
    """Merge LoRA in place and return a normal Action2V state dict."""
    if rank == 0:
        return state_dict

    adapter_suffix = f".lora_A.{adapter_name}.weight"
    scaling = alpha / rank
    adapter_keys = [
        key for key in state_dict if key.endswith(adapter_suffix)
    ]
    if not adapter_keys:
        raise ValueError("LoRA is enabled but the gathered state has no adapters")

    for a_key in adapter_keys:
        stem = a_key[:-len(adapter_suffix)]
        b_key = f"{stem}.lora_B.{adapter_name}.weight"
        base_key = f"{stem}.base_layer.weight"
        if b_key not in state_dict or base_key not in state_dict:
            raise ValueError(f"Incomplete LoRA state for {stem}")
        base = state_dict[base_key]
        delta = torch.mm(
            state_dict[b_key].to(device=device, dtype=torch.float32),
            state_dict[a_key].to(device=device, dtype=torch.float32),
        ).mul_(scaling).to(device=base.device)
        output_key = f"{stem}.weight"
        if output_key in state_dict:
            raise ValueError(f"Duplicate merged checkpoint key: {output_key}")
        # Match PEFT: round the sum to the base dtype, not the delta first.
        base.add_(delta)
        del state_dict[a_key], state_dict[b_key]
        state_dict[output_key] = state_dict.pop(base_key)

    discard_markers = tuple(f".{name}." for name in discard_adapters)
    for key in list(state_dict):
        if ".lora_" in key and any(marker in key for marker in discard_markers):
            del state_dict[key]
            continue
        if ".lora_" in key:
            raise ValueError(f"Unsupported unmerged LoRA parameter: {key}")
        output_key = key.replace(".base_layer.", ".")
        if output_key == key:
            continue
        if output_key in state_dict:
            raise ValueError(f"Duplicate merged checkpoint key: {output_key}")
        state_dict[output_key] = state_dict.pop(key)
    return state_dict


class WanModelWrapper(torch.nn.Module):
    """One native DiT behind shared training, adapter and checkpoint utilities."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._text_guidance = None

    @contextmanager
    def text_guidance(self, negative_context, scale):
        """Use separate text KV for optional CFG in every cached model wrapper."""
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Text guidance scale must be finite and nonnegative')
        if scale != 1.0 and (negative_context is None or not len(negative_context)):
            raise ValueError('Text guidance requires a negative context')
        previous = self._text_guidance
        self._text_guidance = None if scale == 1.0 else (negative_context, float(scale))
        try:
            yield
        finally:
            self._text_guidance = previous

    def guided_cache(self, allocate):
        """Allocate both text branches with identical native sink/window policies."""
        cache = allocate()
        if self._text_guidance is not None:
            negative = allocate()
            cache.streams.update({'text_negative_' + name: layers for name, layers in negative.streams.items()})
            cache.cross.extend(negative.cross)
        return cache

    def guided_forward(self, predict, *args, context, **inputs):
        if self._text_guidance is None:
            return predict(*args, context=context, **inputs)
        from utils import text_cfg

        negative_context, scale = self._text_guidance
        names = [name for name in inputs if name.startswith('text_negative_')]
        if not names or inputs.get('kv_cache') is None or len(negative_context) != len(context):
            raise ValueError('Text-guided forward requires independent KV and matching context batches')
        negative_streams = {name.removeprefix('text_negative_'): inputs.pop(name) for name in names}
        negative_inputs = {**inputs, **negative_streams}
        cross = inputs.get('crossattn_cache')
        if cross is not None:
            layers = len(self.model.blocks)
            if len(cross) != 2 * layers:
                raise ValueError('Text-guided cache requires independent cross-attention caches')
            inputs['crossattn_cache'], negative_inputs['crossattn_cache'] = cross[:layers], cross[layers:]
        positive = predict(*args, context=context, **inputs)
        negative = predict(*args, context=negative_context, **negative_inputs)
        return text_cfg(positive, negative, scale)

    def forward(self, *args, **kwargs):
        from utils import model_autocast
        with model_autocast(self.model):
            result = (self.model(*args, **kwargs) if self._text_guidance is None else
                      self.guided_forward(self.model, *args, **kwargs))
        return result.float() if torch.is_tensor(result) else result


class Action2VModel(WanModelWrapper):
    """Block-causal Wan with named model inputs and a shared rolling sampler."""

    def __init__(self, model_dir, device=None, dtype=torch.float32, *,
                 temporal_layout, local_attn_size, sink_size, transformer_type, base):
        config = dict(transformer_type.load_config(str(model_dir)))
        config.update(base=base, local_attn_size=int(local_attn_size), sink_size=int(sink_size))
        super().__init__(transformer_type.from_config(config, torch_dtype=dtype))
        self.model.temporal_layout = temporal_layout
        if device is not None:
            self.model.to(device=device, dtype=dtype)

    def predict_flow(self, noisy_latent, timestep, context, *, noisy_embedding=None,
                     clean_latent=None, prefix=None, secondary_context=None, prompt_mask=None,
                     kv_cache=None, **inputs):
        if noisy_latent.ndim != 5:
            raise ValueError("Latent must have shape [B,F,C,H,W]")
        batch, frames, _, height, width = noisy_latent.shape
        if timestep.shape != (batch, frames):
            raise ValueError("Timestep must have shape [B,F]")
        if len(context) != batch:
            raise ValueError("Context and latent batch sizes differ")
        if (secondary_context is None) != (prompt_mask is None):
            raise ValueError("Secondary context and target-frame prompt mask must be provided together")
        if secondary_context is not None:
            if len(secondary_context) != batch:
                raise ValueError("Secondary context and latent batch sizes differ")
            if prompt_mask.shape != (batch, frames) or prompt_mask.dtype != torch.bool:
                raise ValueError("Target-frame prompt mask must be boolean [B,F]")
        patch_t, patch_h, patch_w = self.model.patch_size
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError("Latent shape is not divisible by Wan patch size")
        patch_grid = (frames // patch_t, height // patch_h, width // patch_w)
        if noisy_embedding is not None and noisy_embedding.shape != (batch, self.model.dim, *patch_grid):
            raise ValueError("Precomputed noisy embedding has an invalid shape")
        if clean_latent is not None and clean_latent.shape != noisy_latent.shape:
            raise ValueError("Teacher-forcing clean latent must match noisy latent")
        # Control hooks see only prepared conditions. Cache streams and sampler
        # positions pass through unchanged, even when a hook replaces its inputs.
        runtime = {name: inputs.pop(name) for name in (
            *self.model.cache_streams, 'crossattn_cache', 'current_start',
            'position_start', 'extra_sink_size', 'commit_cache',
        ) if name in inputs}
        kwargs = self.model.condition_kwargs(inputs, batch, frames)
        kwargs.update(runtime)
        if kv_cache is not None:
            if prefix is not None or clean_latent is not None:
                raise ValueError("Cached inference does not accept clean context")
            kwargs['kv_cache'] = kv_cache
        else:
            kwargs.update(clean_x=None if clean_latent is None else clean_latent.transpose(1, 2),
                          aug_t=None if clean_latent is None else torch.zeros_like(timestep))
            if prefix is not None:
                kwargs.update(self.model.condition_kwargs(
                    prefix.inputs.forward_kwargs(), batch, prefix.latent.shape[1], prefix=True))
                kwargs['prefix_embeddings'] = prefix.embedding
        flow = self.model(
            noisy_latent.transpose(1, 2), t=timestep, context=context, seq_len=math.prod(patch_grid),
            patch_embeddings=noisy_embedding, secondary_context=secondary_context,
            prompt_mask=prompt_mask, **kwargs,
        )
        return flow.transpose(1, 2)

    def new_cache(self, batch_size, frames, height, width, device, dtype,
                  extra_sink_size=0, capacity_frames=None, sink_frames=None):
        """Allocate the named streams declared by the native model."""
        from wan.kv_cache.causal import make_causal_cache
        def allocate():
            return make_causal_cache(self.model, batch_size, frames, height, width, device, dtype,
                                     extra_sink_size=extra_sink_size, capacity_frames=capacity_frames,
                                     sink_frames=sink_frames)
        return self.guided_cache(allocate)

    @torch.no_grad()
    def sample_causal_sequence(self, initial_noise, context, inputs, scheduler, *, prefix=None,
                               noise_generators=None, noise_scale=1.0, negative_context=None,
                               guidance_scale=1.0, scheduler_factory=None, sink_frames=None):
        """Run few-step sampling or CFG UniPC with opaque temporal conditions.

        Initial noise already uses noise_scale; additional transition draws use
        the same scale. Model inputs are sliced without materializing copies.
        """
        from sampling import denoise_few_step_block, sample_causal_sequence

        if initial_noise.ndim != 5:
            raise ValueError("Noise must have shape [B,F,C,H,W]")
        batch, frames, _, height, width = initial_noise.shape
        block = self.model.num_frame_per_block
        tokens_per_frame = (height // self.model.patch_size[1]) * (width // self.model.patch_size[2])
        if len(context) != batch:
            raise ValueError("Prompt and latent batch sizes differ")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Guidance scale must be finite and non-negative")
        if guidance_scale != 1.0 and negative_context is None:
            raise ValueError("Text guidance requires a negative context")
        guided = negative_context is not None
        if guided and len(negative_context) != batch:
            raise ValueError("Negative prompt and latent batch sizes differ")
        if scheduler_factory is None and scheduler is None:
            raise ValueError("Causal sampling requires a scheduler or factory")
        if not math.isfinite(noise_scale) or noise_scale <= 0:
            raise ValueError("Noise scale must be finite and positive")
        if noise_generators is not None and len(noise_generators) != batch:
            raise ValueError("Noise generators and latent batch sizes differ")
        prefix_frames = 0 if prefix is None else prefix.latent.shape[1]
        layout = self.model.temporal_layout
        generated_prefix = layout.generated_prefix_frames(prefix_frames)
        prefilled_frames = prefix_frames - generated_prefix
        total_frames = frames + prefilled_frames
        prefix_blocks = layout.prefix_blocks(prefix_frames)
        extra_sink_size = layout.extra_sink_frames(prefix_frames)
        target_blocks = layout.target_blocks(frames - generated_prefix, prefix_frames=prefix_frames)

        def new_cache():
            return self.new_cache(batch, total_frames, height, width,
                                  initial_noise.device, initial_noise.dtype, extra_sink_size,
                                  sink_frames=sink_frames)

        cache = new_cache()
        negative_cache = new_cache() if guided else None

        def draw_noise(reference):
            if noise_generators is None:
                noise = torch.randn_like(reference)
            else:
                noise = torch.stack([torch.randn(reference.shape[1:], generator=generator,
                                                 device=reference.device, dtype=reference.dtype)
                                     for generator in noise_generators])
            return noise.mul_(noise_scale)

        def predict_flow(latent, timestep, embedding, controls, frame, prompt_context, active_cache, *, commit=False):
            position = frame * tokens_per_frame
            return self.predict_flow(
                latent, timestep, prompt_context, noisy_embedding=embedding,
                **controls.forward_kwargs(), **active_cache.forward_kwargs(),
                current_start=position, position_start=position, extra_sink_size=extra_sink_size,
                commit_cache=commit,
            )

        if prefix_frames:
            if prefix.latent.shape != (batch, prefix_frames, initial_noise.shape[2], height, width):
                raise ValueError("Prefix latent does not match the target layout")
            if prefix.embedding.shape != (batch, self.model.dim, prefix_frames,
                                           height // self.model.patch_size[1], width // self.model.patch_size[2]):
                raise ValueError("Prefix embedding does not match the target grid")

        def prepare_block(start, length):
            prefill = start < prefilled_frames
            controls = (prefix.inputs.slice(start, length) if prefill
                        else inputs.slice(start - prefilled_frames, length))

            def restore(current):
                if generated_prefix and start == 0:
                    return torch.cat((prefix.latent, current[:, generated_prefix:]), dim=1)
                return current

            def predict_block(current, timestep_value):
                current = restore(current)
                timestep = timestep_value.float().expand(batch, length).clone()
                if generated_prefix and start == 0:
                    timestep[:, :generated_prefix] = 0
                embedding = self.embed_latent(current)
                flow = predict_flow(current, timestep, embedding, controls, start, context, cache)
                if guided:
                    negative_flow = predict_flow(current, timestep, embedding, controls,
                                                 start, negative_context, negative_cache)
                    from utils import text_cfg
                    flow = text_cfg(flow, negative_flow, guidance_scale)
                return flow

            def commit(clean):
                clean = restore(clean)
                timestep = torch.full((batch, length), getattr(self.model, "context_timestep", 0.0),
                                      device=clean.device, dtype=torch.float32)
                embedding = prefix.embedding[:, :, start:start + length] if prefill else self.embed_latent(clean)
                predict_flow(clean, timestep, embedding, controls, start, context, cache, commit=True)
                if guided:
                    predict_flow(clean, timestep, embedding, controls, start, negative_context, negative_cache, commit=True)

            predict_block.restore = restore
            return predict_block, commit

        def denoise(latent, predict):
            latent = predict.restore(latent)
            block_scheduler = scheduler_factory() if scheduler_factory is not None else scheduler
            return denoise_few_step_block(latent, block_scheduler, predict, draw_noise,
                                          restore=predict.restore)

        return sample_causal_sequence(
            initial_noise, target_blocks, prepare_block, denoise,
            prefix=None if prefix is None or generated_prefix else prefix.latent, prefix_lengths=prefix_blocks,
        )

    def embed_latent(self, latent):
        if latent.ndim != 5:
            raise ValueError("Latent must have shape [B,F,C,H,W]")
        return self.model.patch_embedding(latent.transpose(1, 2))

    def prepare_condition(self, condition_latent, inputs):
        """Embed a clean prefix, preserving its named controls for SFT and prefill."""
        from wan.conditioning import CausalPrefix
        if condition_latent is None:
            if inputs is not None:
                raise ValueError("Prefix inputs require condition latents")
            return None
        if condition_latent.ndim != 5 or condition_latent.shape[1] == 0:
            raise ValueError("Condition latent must have nonempty shape [B,P,C,H,W]")
        batch, frames = condition_latent.shape[:2]
        self.model.temporal_layout.prefix_blocks(frames)
        self.model.condition_kwargs(inputs.forward_kwargs(), batch, frames)
        embedding = self.embed_latent(condition_latent)
        if embedding.shape[2] != frames:
            raise ValueError("Condition temporal patching changed its frame count")
        return CausalPrefix(condition_latent, embedding, inputs)

    def forward(self, noisy_latent, timestep, context, *, clean_latent=None,
                condition_latent=None, condition_inputs=None, secondary_context=None,
                prompt_mask=None, kv_cache=None, **inputs):
        if kv_cache is not None:
            if any(value is not None for value in (clean_latent, condition_latent, condition_inputs,
                                                   secondary_context, prompt_mask)):
                raise ValueError("Cached inference does not accept clean context")
            prefix = None
        else:
            prefix = self.prepare_condition(condition_latent, condition_inputs)
        if self._text_guidance is not None and inputs.get('noisy_embedding') is None:
            inputs['noisy_embedding'] = self.embed_latent(noisy_latent)
        return self.guided_forward(self.predict_flow, noisy_latent, timestep, context=context,
                                   clean_latent=clean_latent, prefix=prefix,
                                   secondary_context=secondary_context, prompt_mask=prompt_mask,
                                   kv_cache=kv_cache, **inputs)




def _validate_project_checkpoint(payload):
    if not isinstance(payload, dict) or set(payload) != {"config", "model"}:
        raise ValueError("Expected checkpoint fields: config, model")
    if not isinstance(payload["config"], dict):
        raise ValueError("Checkpoint config must be a mapping")
    state_dict = payload["model"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint model must be a non-empty state dict")
    return payload, normalize_checkpoint_keys(state_dict)


def read_checkpoint(path):
    """Read project metadata and normalized weights, or native generator weights."""
    payload = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True
    )
    if isinstance(payload, dict) and set(payload) == {"config", "model"}:
        return _validate_project_checkpoint(payload)
    if not isinstance(payload, dict):
        raise ValueError("Unsupported upstream checkpoint")
    for name in ("generator", "generator_ema", "model"):
        state_dict = payload.get(name)
        if isinstance(state_dict, dict) and state_dict:
            return None, normalize_checkpoint_keys(state_dict)
    raise ValueError("Upstream checkpoint has no generator weights")


def checkpoint_config(path):
    """Return project checkpoint metadata, or ``None`` upstream."""
    payload, _ = read_checkpoint(path)
    return None if payload is None else payload["config"]


def _load_upstream(model, state_dict):
    prefix = "model."
    if not state_dict or any(not key.startswith(prefix) for key in state_dict):
        raise ValueError("Upstream generator keys do not match Action2V")
    model.model.load_state_dict(
        {key[len(prefix):]: value for key, value in state_dict.items()},
        strict=True,
    )


def load_checkpoint(model, path):
    """Load project weights or upstream weights, preferring generator over EMA."""
    payload, state_dict = read_checkpoint(path)
    if payload is None:
        _load_upstream(model, state_dict)
    else:
        model.load_state_dict(state_dict, strict=True)


def safetensors_files(checkpoint):
    """Resolve one native file or the standard Diffusers indexed shard layout."""
    directory = Path(checkpoint)
    if directory.suffix == '.safetensors':
        if not directory.is_file():
            raise FileNotFoundError(directory)
        return [directory]
    index = directory / 'diffusion_pytorch_model.safetensors.index.json'
    if index.is_file():
        mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Safetensors index must contain a nonempty weight map')
        if any(not isinstance(name, str) or Path(name).name != name for name in mapping.values()):
            raise ValueError('Safetensors shard filenames must be local to the model directory')
        names = sorted(set(mapping.values()))
    else:
        names = ['diffusion_pytorch_model.safetensors']
    paths = [directory / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing model weights: ' + ', '.join(missing))
    return paths


@torch.no_grad()
def load_safetensors_weights(model, directory):
    """Strictly load native tensors without constructing a second full model."""
    from safetensors import safe_open

    paths, target, found = safetensors_files(directory), model.state_dict(), set()
    for path in paths:
        with safe_open(path, framework='pt', device='cpu') as source:
            for name in source.keys():
                if name in found or name not in target:
                    raise ValueError(f'Unexpected or duplicate native weight: {name}')
                if tuple(source.get_slice(name).get_shape()) != tuple(target[name].shape):
                    raise ValueError(f'Native weight shape mismatch: {name}')
                found.add(name)
    missing = set(target) - found
    if missing:
        raise ValueError(f'Missing native weights: {sorted(missing)[:16]}')
    for path in paths:
        with safe_open(path, framework='pt', device='cpu') as source:
            for name in source.keys():
                target[name].copy_(source.get_tensor(name))


@torch.no_grad()
def decode_latents(vae, latents, *, decoder=None):
    """Decode [B,F,C,H,W] or [F,C,H,W] latents to matching video layouts."""
    unbatched = latents.ndim == 4
    if unbatched:
        latents = latents.unsqueeze(0)
    if latents.ndim != 5:
        raise ValueError(
            f"Expected [B,F,C,H,W] or [F,C,H,W], got {latents.shape}"
        )
    channel_first = latents.permute(0, 2, 1, 3, 4).to(dtype=vae.dtype)
    videos = torch.stack((decoder or vae.decode)(list(channel_first)))
    videos = videos.permute(0, 2, 1, 3, 4)
    return videos[0] if unbatched else videos


def decoded_frame_count(latent_frames):
    if latent_frames <= 0:
        raise ValueError("Latent sequence must contain at least one frame")
    return 1 + VAE_TEMPORAL_STRIDE * (latent_frames - 1)

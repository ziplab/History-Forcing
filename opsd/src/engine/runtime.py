"""Distributed runtime primitives used by training engines."""

import gc
import json
import os
import pickle
import random
import re
import shutil
import time
import traceback
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map_only
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from utils import inference_rng

DISTRIBUTED_TIMEOUT = timedelta(minutes=10)
TRAINER_STATE_NAME = "trainer.pt"
ADAPTER_MODEL_NAME = "adapter_model.pt"
CHECKPOINT_CONFIG_NAME = "config.json"
CHECKPOINT_COMPLETE_NAME = "checkpoint_complete.json"
CHECKPOINT_WRITING_NAME = ".checkpoint-writing"


@dataclass(frozen=True)
class DataLoaderInfo:
    micro_batches: int
    samples_per_pass: int
    repeated_samples_per_pass: int
    steps_per_pass: int


class StepDistributedSampler(Sampler):
    """Repeat data to form complete distributed optimizer steps."""

    def __init__(
        self, dataset, batch_size, gradient_accumulation, num_replicas,
        rank, seed, *, group_ids=None,
    ):
        if len(dataset) == 0:
            raise ValueError("Training dataset cannot be empty")
        if batch_size <= 0 or gradient_accumulation <= 0:
            raise ValueError(
                "Batch size and gradient accumulation must be positive"
            )
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed sampler rank")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.global_batch_size = num_replicas * batch_size
        self.groups = None
        size = len(dataset)
        if group_ids is not None:
            if len(group_ids) != len(dataset):
                raise ValueError('Sampler groups must cover every dataset row')
            groups = {}
            for index, group in enumerate(group_ids):
                groups.setdefault(group, []).append(index)
            self.groups = tuple(groups.values())
            size = sum((len(group) + self.global_batch_size - 1) // self.global_batch_size
                       * self.global_batch_size for group in self.groups)
        global_step_samples = (
            num_replicas * batch_size * gradient_accumulation
        )
        self.total_size = (
            (size + global_step_samples - 1) // global_step_samples
            * global_step_samples
        )
        self.num_samples = self.total_size // num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.groups is None:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            # A whole global microbatch shares a rollout layout. This keeps
            # FSDP forward/backward collectives aligned across ranks.
            batches = []
            width = self.global_batch_size
            for group in self.groups:
                shuffled = [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
                padded = ((len(shuffled) + width - 1) // width) * width
                repeats, remainder = divmod(padded, len(shuffled))
                shuffled = shuffled * repeats + shuffled[:remainder]
                batches.extend(shuffled[start:start + width] for start in range(0, padded, width))
            order = torch.randperm(len(batches), generator=generator).tolist()
            indices = [index for batch in order for index in batches[batch]]
        repeats, remainder = divmod(self.total_size, len(indices))
        indices = indices * repeats + indices[:remainder]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("Distributed sampler produced an invalid shard")
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_lr_scheduler(optimizer, warmup_steps):
    """Constant learning rate with the linear warmup used by video LoRA trainers."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0
        ),
    )


def optimizer_parameter_groups(model, weight_decay):
    """Honor each parameter's no-weight-decay flag with one learning rate."""
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        target = (
            no_decay
            if getattr(parameter, "_no_weight_decay", False)
            else decay
        )
        target.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("The model has no trainable parameters")
    return groups


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return trainable, total


def init_distributed():
    if "RANK" not in os.environ:
        raise RuntimeError("Launch training with torchrun, including for one GPU")
    os.environ.setdefault("NCCL_CROSS_NIC", "1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl", timeout=DISTRIBUTED_TIMEOUT, device_id=device
    )
    return dist.get_rank(), dist.get_world_size(), device


def create_checkpoint_group(world_size):
    """Keep slow rank-0 filesystem writes outside the NCCL watchdog."""
    if world_size == 1:
        return None
    return dist.new_group(backend="gloo", timeout=timedelta(hours=24))


def _destroy_training_group(group=None):
    if dist.is_initialized():
        dist.destroy_process_group(group)


@contextmanager
def training_session(seed):
    """Own one training invocation, including cleanup on setup errors or Ctrl-C.

    Resources register local cleanup before setup errors are shared across ranks.
    No barrier is added on exit; a failed peer may no longer enter collectives.
    """
    if dist.is_initialized():
        raise RuntimeError("Training requires its own process group; a group is already initialized")
    with ExitStack() as resources:
        resources.callback(_destroy_training_group)
        rank, world_size, device = init_distributed()
        configure_runtime(seed, rank)
        checkpoint_group = create_checkpoint_group(world_size)
        if checkpoint_group is not None:
            resources.callback(_destroy_training_group, checkpoint_group)
        yield rank, world_size, device, checkpoint_group, resources


def raise_distributed_errors(error, process_group, context):
    """Propagate rank-local Python errors before the next collective."""
    if not dist.is_initialized():
        if error is not None:
            raise RuntimeError(f"{context} failed:\n{error}")
        return
    world_size = dist.get_world_size(group=process_group)
    if world_size == 1:
        if error is not None:
            raise RuntimeError(f"{context} failed:\n{error}")
        return
    errors = [None] * world_size
    dist.all_gather_object(errors, error, group=process_group)
    failures = [
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if failures:
        raise RuntimeError(f"{context} failed on " + "\n".join(failures))


@contextmanager
def distributed_phase(process_group, context):
    """Finish rank-local work or propagate its error before the next collective."""
    error = None
    try:
        yield
    except (Exception, SystemExit) as exception:
        error = traceback.format_exc()
        # Context-manager exit still owns the original traceback. Release
        # unwound tensor/module locals before waiting for peer ranks.
        traceback.clear_frames(exception.__traceback__)
    raise_distributed_errors(error, process_group, context)


def distributed_call(function, process_group, context):
    """Run rank-local setup and propagate its traceback to every rank."""
    with distributed_phase(process_group, context):
        return function()


def step_optimizer(optimizer, scheduler, process_group, context):
    """Finish a local update on every rank before EMA, logging or checkpointing."""
    with distributed_phase(process_group, context):
        try:
            optimizer.step()
            scheduler.step()
        finally:
            optimizer.zero_grad(set_to_none=True)


def configure_runtime(seed, rank):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_model_assets(
    model_dir, checkpoint, model, data_path=None, require_codec=False, *, config=None,
):
    from backbones import load_backend
    from dataset import validate_dataset_path

    if data_path is not None:
        validate_dataset_path(data_path)

    load_backend(model).validate(
        model_dir,
        checkpoint,
        codec=require_codec,
        config=config,
    )




def build_dataloader(
    dataset, batch_size, gradient_accumulation, num_workers, seed, rank,
    world_size, collate_fn=None, *, group_ids=None,
):
    sampler = StepDistributedSampler(
        dataset,
        batch_size,
        gradient_accumulation,
        world_size,
        rank,
        seed,
        group_ids=group_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_fn,
    )
    if len(loader) == 0:
        raise ValueError("Dataset is too small for the requested configuration")
    micro_batches = len(loader)
    if micro_batches % gradient_accumulation:
        raise RuntimeError("Sampler did not form complete optimizer steps")
    info = DataLoaderInfo(
        micro_batches=micro_batches,
        samples_per_pass=sampler.total_size,
        repeated_samples_per_pass=sampler.total_size - len(dataset),
        steps_per_pass=micro_batches // gradient_accumulation,
    )
    return loader, sampler, info


def close_dataloader(loader):
    """Stop persistent workers and their pin-memory thread, as in Lightning reset."""
    iterator, loader._iterator = loader._iterator, None
    if iterator is not None and loader.num_workers > 0:
        iterator._shutdown_workers()


def start_training_pass(loader, sampler, *, data_pass, batch_index, batch_size,
                        text_stores=()):
    """Resume a pass with matching sampler and absolute text/batch cursors."""
    if not 0 <= batch_index < len(loader):
        raise ValueError("Training batch cursor is outside the data pass")
    sampler.set_epoch(data_pass)
    stores = {id(store): store for store in text_stores if store is not None}
    if stores:
        order = list(iter(sampler))
        for store in stores.values():
            store.start_pass(order, batch_size, batch_index)
    iterator = iter(loader)
    for _ in range(batch_index):
        next(iterator)
    return iterator


def accumulation_window(batch_index, info, batch_size, accumulation):
    window_start = batch_index // accumulation * accumulation
    window_end = window_start + accumulation
    if window_end > info.micro_batches:
        raise RuntimeError("Incomplete gradient accumulation window")
    return batch_index + 1 == window_end, accumulation * batch_size


def _trainable_leaf(module):
    """Separate trainable control modules from frozen FSDP flat parameters."""
    parameters = tuple(module.parameters(recurse=False))
    return (not list(module.children()) and bool(parameters)
            and all(parameter.requires_grad for parameter in parameters))


def _fsdp_module_policy(module, recurse, nonwrapped_numel, default):
    """Honor model-declared FSDP boundaries, as in Composer's _fsdp_wrap."""
    if recurse:
        return True
    override = getattr(module, "_fsdp_wrap", None)
    return (bool(override) if override is not None
            else default(module, recurse, nonwrapped_numel))


def _fsdp_input_views(module, args, kwargs):
    """Give frozen units their own input-gradient completion events.

    FSDP waits for all input gradients before resharding frozen parameters.
    A shared conditioning tensor otherwise delays every layer until its total
    gradient is ready. Views separate these events without copying tensor data.
    """
    if torch.is_grad_enabled():
        return tree_map_only(
            torch.Tensor,
            lambda value: value.view_as(value) if value.requires_grad else value,
            (args, kwargs),
        )


def wrap_fsdp(
    module, device, transformer_layer_cls=None, process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
):
    if isinstance(sharding_strategy, str):
        sharding_strategy = ShardingStrategy[sharding_strategy]
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
        cast_forward_inputs=False,
        # Preserve timestep/geometry metadata; autocast selects compute dtypes.
        cast_root_forward_inputs=False,
    )
    if transformer_layer_cls is None:
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=int(5e7)
        )
    else:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(
                transformer_layer_cls
                if isinstance(transformer_layer_cls, tuple)
                else (transformer_layer_cls,)
            ),
        )
    if {parameter.requires_grad for parameter in module.parameters()} == {False, True}:
        auto_wrap_policy = partial(
            _or_policy,
            policies=[
                partial(lambda_auto_wrap_policy, lambda_fn=_trainable_leaf),
                auto_wrap_policy,
            ],
        )
    if any(hasattr(child, "_fsdp_wrap") for child in module.modules()):
        auto_wrap_policy = partial(_fsdp_module_policy, default=auto_wrap_policy)
    wrapped = FullyShardedDataParallel(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        process_group=process_group,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=False,
    )
    for unit in FullyShardedDataParallel.fsdp_modules(wrapped):
        # This is the unit's own flat parameter, excluding nested FSDP units.
        if unit._flat_param is not None and not unit._flat_param.requires_grad:
            unit.register_forward_pre_hook(_fsdp_input_views, with_kwargs=True)
    return wrapped


def _prepare_parameter_efficient_model(module, device):
    """Place a PEFT model without turning frozen weights into grad tensors."""
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not torch.is_floating_point(parameter):
                continue
            # Inactive OPSD adapters are frozen but remain FP32 EMA targets.
            adapter = ".lora_" in f".{name}"
            dtype = (
                torch.float32
                if parameter.requires_grad or adapter
                else torch.bfloat16
            )
            parameter.data = parameter.data.to(dtype=dtype)
    return module.to(device=device)


def wrap_training_model(
    module,
    device,
    parameter_efficient,
    transformer_layer_cls=None,
    process_group=None,
    *,
    sharding_strategy=None,
):
    """Wrap full tuning and explicitly sharded LoRA with FSDP."""
    strategy = (ShardingStrategy[sharding_strategy] if isinstance(sharding_strategy, str)
                else sharding_strategy)
    shard_lora = parameter_efficient and strategy not in (None, ShardingStrategy.NO_SHARD)
    if not parameter_efficient or shard_lora:
        if shard_lora:
            module.float()
            # Match Student/Teacher adapter shard layouts for local EMA updates.
            for name, child in module.named_modules():
                if ('.lora_' in f'.{name}' and not list(child.children())
                        and any(True for _ in child.parameters(recurse=False))):
                    child._fsdp_wrap = True
        return wrap_fsdp(
            module,
            device,
            transformer_layer_cls=transformer_layer_cls,
            process_group=process_group,
            sharding_strategy=strategy if strategy is not None else ShardingStrategy.FULL_SHARD,
        )

    module = _prepare_parameter_efficient_model(module, device)
    world_size = (
        dist.get_world_size(group=process_group)
        if dist.is_initialized()
        else 1
    )
    if world_size == 1:
        return module
    return DistributedDataParallel(
        module,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        process_group=process_group,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def distributed_strategy(model):
    if isinstance(model, FullyShardedDataParallel):
        return "fsdp"
    if isinstance(model, DistributedDataParallel):
        return "ddp"
    return "local"


def unwrap_distributed_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def maybe_no_sync(model, synchronize):
    """Suppress gradient communication for one complete forward/backward."""
    if synchronize or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def training_autocast(*, cache_enabled=True):
    """One BF16 compute policy for full, FSDP and PEFT training."""
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        cache_enabled=cache_enabled,
    )


def clip_grad_norm(model, max_norm):
    if isinstance(model, FullyShardedDataParallel):
        return model.clip_grad_norm_(max_norm)
    parameters = (
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    )
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def load_text_encoder(
    model_dir, device, model, use_fsdp=True, process_group=None,
):
    from backbones import load_backend

    shard_fn = None
    if use_fsdp:
        def shard(module, sync_module_states=False):
            del sync_module_states
            return wrap_fsdp(module, device, process_group=process_group)

        shard_fn = shard
    return load_backend(model).text_encoder(
        model_dir, device, shard_fn=shard_fn
    )


class TextEmbeddingStore:
    """Memory-mapped CPU embeddings with double-buffered GPU batches."""

    def __init__(
        self, sample_prompt_groups, embeddings, device,
    ):
        self.sample_prompt_groups = tuple(
            tuple(group) for group in sample_prompt_groups
        )
        self.embeddings = embeddings
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)
        self.prompt_batches = []
        self.current_index = -1
        self.current = {}
        self.staged = None

    def start_pass(self, sample_indices, batch_size, start_batch_index=0):
        """Build this rank's batches in the exact sampler order."""
        self.copy_stream.synchronize()
        prompt_groups = [
            self.sample_prompt_groups[index] for index in sample_indices
        ]
        self.prompt_batches = [
            list(dict.fromkeys(
                prompt
                for group in prompt_groups[start:start + batch_size]
                for prompt in group
            ))
            for start in range(0, len(prompt_groups), batch_size)
        ]
        first_window = start_batch_index
        if first_window >= len(self.prompt_batches):
            raise ValueError("Text embedding resume position exceeds the pass")
        self.current_index = first_window - 1
        self.staged = None
        self._stage(first_window)

    def _stage(self, index):
        if index >= len(self.prompt_batches):
            self.staged = None
            return
        device_embeddings = {}
        pinned = []
        with torch.cuda.stream(self.copy_stream):
            for prompt in self.prompt_batches[index]:
                if prompt in self.current:
                    device_embedding = self.current[prompt]
                else:
                    host_embedding = self.embeddings[prompt].pin_memory()
                    device_embedding = host_embedding.to(
                        self.device, non_blocking=True
                    )
                    pinned.append(host_embedding)
                device_embeddings[prompt] = device_embedding
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)
        self.staged = (index, device_embeddings, ready, pinned)

    def _activate(self, index):
        if self.staged is None or self.staged[0] != index:
            raise RuntimeError("Text embedding prefetch is out of order")
        _, device_embeddings, ready, _ = self.staged
        ready.synchronize()
        self.current = device_embeddings
        self.current_index = index
        self._stage(index + 1)

    def encode(self, prompts, batch_index):
        window_index = batch_index
        if not 0 <= window_index < len(self.prompt_batches):
            raise ValueError("Text embedding batch index is outside the data pass")
        if window_index != self.current_index:
            if self.staged is None or self.staged[0] != window_index:
                # Alternating models can skip windows. Finish the old host
                # transfer before replacing its pinned buffers and prefetch.
                self.copy_stream.synchronize()
                self._stage(window_index)
            self._activate(window_index)
        missing = set(prompts).difference(self.current)
        if missing:
            raise RuntimeError(
                f"GPU text window is missing {len(missing)} prompt(s)"
            )
        current_stream = torch.cuda.current_stream(self.device)
        context = [self.current[prompt] for prompt in prompts]
        for prompt in dict.fromkeys(prompts):
            self.current[prompt].record_stream(current_stream)
        return context


def precompute_text_embeddings(
    model_dir, prompt_groups, batch_size, output_dir, device, rank,
    world_size, sample_indices, process_group=None,
    error_group=None, model=None,
):
    """Encode prompt shards on all ranks, memory-map them, then release T5."""
    def prepare_prompts():
        groups = [tuple(group) for group in prompt_groups]
        if any(not group or any(not isinstance(prompt, str) for prompt in group) for group in groups):
            raise ValueError("Every sample text group must contain strings")
        unique = list(dict.fromkeys(prompt for group in groups for prompt in group))
        rank_prompts = list(dict.fromkeys(
            prompt
            for index in sample_indices
            for prompt in groups[index]
        ))
        return groups, unique, rank_prompts

    prompt_groups, unique_prompts, rank_prompts = distributed_call(
        prepare_prompts, error_group, "Text embedding cache preparation",
    )
    store_dir = output_dir / "text_embeddings"

    def write_metadata():
        if rank == 0:
            print(
                f"Precomputing {len(unique_prompts)} unique text embeddings "
                f"across {world_size} rank(s)",
                flush=True,
            )
            store_dir.mkdir(parents=True, exist_ok=True)
            save_torch_state(store_dir / "metadata.pt", {"sample_prompt_groups": prompt_groups})

    distributed_call(write_metadata, error_group, "Text embedding metadata write")

    def encode_shard():
        embeddings = {}
        if rank_prompts:
            from backbones import load_backend

            backend = load_backend(model)
            text_encoder = load_text_encoder(
                model_dir,
                device,
                model,
                use_fsdp=False,
            )
            if any(
                parameter.requires_grad
                for parameter in getattr(
                    text_encoder, "model", text_encoder
                ).parameters()
            ):
                raise ValueError(
                    "Text embeddings can only be cached for a frozen encoder"
                )
            with torch.no_grad():
                for start in range(0, len(rank_prompts), batch_size):
                    prompt_batch = rank_prompts[start:start + batch_size]
                    output = backend.encode_context(
                        prompt_batch, device, text_encoder
                    )
                    try:
                        embeddings.update({
                            prompt: embedding.detach().to("cpu", copy=True)
                            for prompt, embedding in zip(
                                prompt_batch, output, strict=True
                            )
                        })
                    finally:
                        del output
            torch.cuda.synchronize(device)
        return embeddings

    try:
        embeddings = distributed_call(encode_shard, error_group, "Text embedding precomputation")
    finally:
        if rank_prompts:
            gc.collect()
            torch.cuda.empty_cache()

    local_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in embeddings.values()
    )
    shard_path = store_dir / f"rank-{rank:05d}-of-{world_size:05d}.pt"
    def write_shard():
        save_torch_state(
            shard_path,
            {
                "embeddings": embeddings,
                "count": torch.tensor(len(embeddings), dtype=torch.int64),
            },
        )

    distributed_call(write_shard, error_group, "Text embedding shard write")

    total_bytes = torch.tensor(local_bytes, device=device, dtype=torch.int64)
    dist.all_reduce(total_bytes, group=process_group)
    embeddings = None

    def load_cache():
        metadata = torch.load(
            store_dir / "metadata.pt", map_location="cpu", weights_only=True
        )
        embeddings = {}
        for shard_rank in range(world_size):
            shard = torch.load(
                store_dir
                / f"rank-{shard_rank:05d}-of-{world_size:05d}.pt",
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            shard_embeddings = shard["embeddings"]
            if shard["count"].item() != len(shard_embeddings):
                raise RuntimeError(
                    f"Invalid text embedding shard {shard_rank}"
                )
            embeddings.update(shard_embeddings)
        if len(embeddings) != len(unique_prompts):
            raise RuntimeError(
                "Text embedding shards contain "
                f"{len(embeddings)}/{len(unique_prompts)} unique prompts"
            )
        return TextEmbeddingStore(
            metadata["sample_prompt_groups"],
            embeddings,
            device,
        )

    embedding_store = distributed_call(load_cache, error_group, "Text embedding cache load")
    if rank == 0:
        cached_mib = total_bytes.item() / 2**20
        print(
            f"Memory-mapped text embeddings ({cached_mib:.1f} MiB); "
            "unloaded text encoder",
            flush=True,
        )
    return embedding_store


def distributed_mean(
    value_sum, sample_count, world_size, process_group=None,
):
    """Global mean, weighted by the actual number of samples on each rank."""
    count = torch.tensor(
        sample_count, dtype=torch.float32, device=value_sum.device
    )
    if world_size > 1:
        dist.all_reduce(
            value_sum, op=dist.ReduceOp.SUM, group=process_group
        )
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=process_group)
    return value_sum / count


def require_finite(values, context, *, optimizers=(), process_group=None):
    """Make every rank stop before backward or an update if any value is invalid."""
    tensors = {
        name: value.detach() if torch.is_tensor(value) else value
        for name, value in values.items()
    }
    device = next((value.device for value in tensors.values()
                   if torch.is_tensor(value)), torch.device("cpu"))
    finite = torch.stack([
        torch.isfinite(torch.as_tensor(value, device=device)).all()
        for value in tensors.values()
    ]).all().to(torch.int32)
    if dist.is_initialized() and dist.get_world_size(group=process_group) > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=process_group)
    if finite.item():
        return
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    invalid = {
        name: torch.as_tensor(value).float().cpu().tolist()
        for name, value in tensors.items()
        if not torch.isfinite(torch.as_tensor(value)).all().item()
    }
    raise FloatingPointError(f"Non-finite {context}; local invalid values={invalid}")


class ExperimentTracker:
    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.finish()

    def __init__(self, args, config):
        self.run = None
        self.metrics_path = args.output_dir / "metrics.jsonl"
        # Scalar-only SDK work must not advance the training RNG or touch GPUs.
        with inference_rng(device="cpu"):
            if args.report_to == "wandb":
                import wandb

                self.run = wandb.init(
                    project=args.project_name,
                    name=args.run_name,
                    config=config,
                    dir=str(args.output_dir),
                    mode=args.tracker_mode,
                )
            elif args.report_to == "swanlab":
                import swanlab

                self.run = swanlab.init(
                    project=args.project_name,
                    experiment_name=args.run_name,
                    config=config,
                    logdir=str(args.output_dir / "swanlog"),
                    mode=args.tracker_mode,
                )

    def log(self, metrics, step):
        with self.metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.run is not None:
            with inference_rng(device="cpu"):
                self.run.log(metrics, step=step)

    def finish(self):
        run, self.run = self.run, None
        if run is not None:
            with inference_rng(device="cpu"):
                run.finish()


def full_state_dict(model):
    if isinstance(model, FullyShardedDataParallel):
        policy = FullStateDictConfig(
            offload_to_cpu=next(model.parameters()).device.type != 'cpu', rank0_only=True,
        )
        with FullyShardedDataParallel.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, policy
        ):
            return model.state_dict()
    if dist.is_initialized() and dist.get_rank() != 0:
        return {}
    state = unwrap_distributed_model(model).state_dict()
    return {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in state.items()
    }


def save_torch_state(path, state):
    """Publish a serialized state atomically and remove failed temporary files."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_training_model(directory, model, backend, model_config, base_checkpoint,
                          identity, lora_rank, lora_alpha, rank, device, group,
                          *, artifacts=None, merge=None):
    """Gather/export once; recipes may supply extra artifacts and a merge policy."""
    state = distributed_call(
        lambda: backend.normalize_state_dict(full_state_dict(model)),
        group, "Training model state gather",
    )

    def write():
        nonlocal state
        # The write frame owns the gather; error consensus clears that frame.
        export_state, state = state, None
        if rank != 0:
            return
        directory.mkdir(parents=True, exist_ok=True)
        if artifacts is not None:
            for filename, artifact in artifacts(export_state):
                save_torch_state(directory / filename, artifact)
        elif lora_rank:
            save_adapter_model(
                directory / ADAPTER_MODEL_NAME,
                backend.lora_state_dict(backend.unwrap(model), export_state),
            )
        else:
            (directory / ADAPTER_MODEL_NAME).unlink(missing_ok=True)
        merge_state = backend.merge_lora if merge is None else merge
        merged = merge_state(export_state, lora_rank, lora_alpha, device)
        save_torch_state(directory / "model.pt", backend.checkpoint(merged, config=model_config))
        save_json_state(
            directory / CHECKPOINT_CONFIG_NAME,
            build_checkpoint_config(base_checkpoint, identity),
        )

    distributed_call(write, group, "Training model checkpoint export")


def full_optimizer_state_dict(model, optimizer):
    """Return a rank-0 CPU optimizer state for FSDP, DDP, or local training."""
    if isinstance(model, FullyShardedDataParallel):
        offload = next(model.parameters()).device.type != 'cpu'
        with FullyShardedDataParallel.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=offload, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=offload, rank0_only=True),
        ):
            return FullyShardedDataParallel.optim_state_dict(model, optimizer)
    if dist.is_initialized() and dist.get_rank() != 0:
        return None

    def to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(to_cpu(item) for item in value)
        return value

    return to_cpu(optimizer.state_dict())


def restore_optimizer_state_dict(
    model,
    optimizer,
    state,
    rank,
    checkpoint_dir,
):
    """Restore an optimizer state for FSDP or replicated training."""
    if isinstance(model, FullyShardedDataParallel):
        sharded = FullyShardedDataParallel.scatter_full_optim_state_dict(
            state if rank == 0 else None,
            model,
            optim=optimizer,
        )
        optimizer.load_state_dict(sharded)
        return

    if rank != 0:
        payload = torch.load(
            checkpoint_dir / TRAINER_STATE_NAME,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = payload["optimizer"]
    optimizer.load_state_dict(state)


def save_adapter_model(path, adapter):
    if not adapter:
        raise ValueError("Adapter state dict cannot be empty")
    save_torch_state(path, adapter)


def build_checkpoint_identity(
    model, lora_rank, lora_alpha, lora_dropout, lora_target_modules,
    training_config, *, freeze_camera=False,
):
    training_config = dict(training_config)
    if freeze_camera:
        training_config["freeze_camera"] = True
    lora = None
    if lora_rank:
        lora = {
            "rank": int(lora_rank),
            "alpha": float(lora_alpha),
            "dropout": float(lora_dropout),
            "target_modules": list(lora_target_modules),
        }
    return {
        "model": {"backend": model},
        "lora": lora,
        "training": training_config,
    }


def common_training_identity(args, backend):
    """Optimizer/data settings and the backend's native runtime policy."""
    settings = {name: getattr(args, name) for name in (
        "batch_size", "gradient_accumulation", "seed", "learning_rate",
        "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
        "max_grad_norm", "lr_warmup_steps",
    )}
    settings["native_model"] = backend.training_identity()
    return settings


def build_checkpoint_config(base_checkpoint, identity):
    config = dict(identity)
    config["model"] = {
        **identity["model"],
        "base_checkpoint": str(Path(base_checkpoint).absolute()),
    }
    return config


def save_json_state(path, state):
    """Publish one UTF-8 JSON record and remove partial temporary writes."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def begin_checkpoint_write(output_dir, step, rank, process_group=None):
    """Invalidate a prior save before replacing its individually atomic files."""
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    def initialize():
        if rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / CHECKPOINT_WRITING_NAME).touch()
            for name in (
                TRAINER_STATE_NAME,
                ADAPTER_MODEL_NAME,
                CHECKPOINT_CONFIG_NAME,
                CHECKPOINT_COMPLETE_NAME,
                "model.pt",
                "student_adapter.pt",
                "teacher_adapter.pt",
                "teacher_ema.pt",
                "generator_ema.pt",
                "evaluation.json",
            ):
                path = checkpoint_dir / name
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
    distributed_call(initialize, process_group, "Checkpoint initialization")
    return checkpoint_dir


def complete_checkpoint_write(directory, rank, world_size, process_group=None,
                              *, extra_files=()):
    """Publish completion only after all models, trainer states and RNGs exist."""
    def finish():
        if rank != 0:
            return
        required = ("model.pt", CHECKPOINT_CONFIG_NAME, TRAINER_STATE_NAME,
                    *(rng_state_path(Path(), index, world_size).name
                      for index in range(world_size)), *extra_files)
        sizes = {}
        for name in required:
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Checkpoint is incomplete: {path}")
            sizes[name] = path.stat().st_size
        save_json_state(directory / CHECKPOINT_COMPLETE_NAME,
                               {"files": sizes})
        (directory / CHECKPOINT_WRITING_NAME).unlink(missing_ok=True)

    distributed_call(finish, process_group, "Checkpoint completion")


def checkpoint_is_complete(directory):
    """Accept marked saves and the complete file layout used before markers."""
    if (directory / CHECKPOINT_WRITING_NAME).exists():
        return False
    marker = directory / CHECKPOINT_COMPLETE_NAME
    if marker.is_file():
        try:
            files = json.loads(marker.read_text(encoding="utf-8"))["files"]
            required = {"model.pt", CHECKPOINT_CONFIG_NAME, TRAINER_STATE_NAME}
            return isinstance(files, dict) and required <= files.keys() and all(
                (directory / name).is_file()
                and (directory / name).stat().st_size == size and size > 0
                for name, size in files.items()
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False
    # Older engines wrote trainer.pt last. Require its preceding model/config
    # and RNG files; DMD also needs its second trainer and optional EMA.
    try:
        config = load_checkpoint_config(directory / "model.pt")
        world_size = _read_trainer_state(directory / TRAINER_STATE_NAME)["world_size"]
        required = ["model.pt", CHECKPOINT_CONFIG_NAME, TRAINER_STATE_NAME]
        training = config["training"]
        is_opsd = training.get("trainer") in ("opsd", "opsd-i2v")
        if config["lora"] is not None:
            required += (["student_adapter.pt", "teacher_adapter.pt"]
                         if is_opsd
                         else [ADAPTER_MODEL_NAME])
        elif is_opsd:
            required.append("teacher_ema.pt")
        if training.get("recipe") == "dmd" and directory.name != "fake_score":
            if not checkpoint_is_complete(directory / "fake_score"):
                return False
            step = re.fullmatch(r"checkpoint-(\d+)", directory.name)
            if training.get("ema_decay") and (step is None or
                    int(step.group(1)) >= training.get("ema_start_step", 0)):
                required.append("generator_ema.pt")
        required.extend(rng_state_path(directory, rank, world_size).name for rank in range(world_size))
        return all(
            (directory / name).is_file() and (directory / name).stat().st_size > 0
            for name in required
        )
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError):
        return False


def prune_checkpoints(output_dir, max_checkpoints, latest_checkpoint=None):
    """Limit completed saves, always retaining the checkpoint just published."""
    if max_checkpoints == 0:
        return
    if max_checkpoints < 0:
        raise ValueError("max_checkpoints cannot be negative")
    output_dir = Path(output_dir).resolve()
    latest_checkpoint = None if latest_checkpoint is None else Path(latest_checkpoint).resolve()
    checkpoints = [
        (int(match.group(1)), path)
        for path in output_dir.iterdir()
        if path.is_dir() and not path.is_symlink()
        if (match := re.fullmatch(r"checkpoint-(\d{6,})", path.name))
        if checkpoint_is_complete(path)
    ]
    checkpoints.sort(key=lambda item: (item[1] == latest_checkpoint, item[0]))
    for _, path in checkpoints[:-max_checkpoints]:
        if path.resolve().parent != output_dir:
            raise ValueError(f"Refusing to remove checkpoint outside {output_dir}: {path}")
        shutil.rmtree(path)


def write_training_checkpoint(output_dir, step, *, rank, world_size,
                              process_group, write_payload, max_checkpoints,
                              evaluate=None):
    """Publish a recipe save before retention and live-model evaluation.

    All ranks execute write_payload in the same order. Its collective/model
    writes retain their own distributed error propagation and return the extra
    files required in addition to the main model, trainer and rank RNG states.
    """
    started = time.monotonic()
    directory = begin_checkpoint_write(output_dir, step, rank, process_group)
    extra_files = write_payload(directory)
    complete_checkpoint_write(
        directory, rank, world_size, process_group, extra_files=extra_files,
    )
    distributed_call(
        lambda: prune_checkpoints(output_dir, max_checkpoints, directory)
        if rank == 0 else None,
        process_group, "Training checkpoint retention",
    )
    if rank == 0:
        tqdm.write(f"Checkpoint {step:06d} complete in {time.monotonic() - started:.1f}s: {directory}")
    if evaluate is not None:
        evaluate(directory, step)
    return directory


def load_checkpoint_config(checkpoint):
    path = checkpoint.with_name(CHECKPOINT_CONFIG_NAME)
    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load checkpoint config {path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    required = {"model", "lora", "training"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Invalid checkpoint config: {path}")
    model = config["model"]
    model_fields = {"backend", "base_checkpoint"}
    if not isinstance(model, dict) or not model_fields <= model.keys():
        raise ValueError(f"Checkpoint config has an invalid model section: {path}")
    config['model'] = model = {key: model[key] for key in model_fields}
    if not all(
        isinstance(model[name], str)
        for name in ("backend", "base_checkpoint")
    ):
        raise ValueError(f"Checkpoint config has invalid model paths: {path}")
    lora = config["lora"]
    if lora is not None and (
        not isinstance(lora, dict)
        or set(lora) != {"rank", "alpha", "dropout", "target_modules"}
        or type(lora["rank"]) is not int
        or lora["rank"] <= 0
    ):
        raise ValueError(f"Checkpoint config has an invalid LoRA section: {path}")
    if not isinstance(config["training"], dict):
        raise ValueError(f"Checkpoint config has invalid training settings: {path}")
    # Normalize the v3 compact-RoPE checkpoint schema to identity v4.
    training = config['training']
    teacher_kv = training.get('teacher_kv', {})
    if (training.get('trainer') == 'opsd-i2v' and training.get('identity_version') == 3
            and isinstance(teacher_kv, dict) and teacher_kv.get('rope_policy') == 'compact'):
        teacher_kv.pop('rope_policy')
        training['identity_version'] = 4
    return config


def _configuration_differences(saved, current, prefix=""):
    """Report differing field paths without dumping prompts or configuration values."""
    if isinstance(saved, dict) and isinstance(current, dict):
        for name in sorted(saved.keys() | current.keys()):
            path = f"{prefix}.{name}" if prefix else name
            if name not in saved or name not in current:
                yield path
            else:
                yield from _configuration_differences(saved[name], current[name], path)
    elif saved != current:
        yield prefix


def validate_checkpoint_identity(
    checkpoint, identity, model_config, saved_model_config,
):
    config = load_checkpoint_config(checkpoint)
    saved = {
        "model": {"backend": config["model"]["backend"]},
        "lora": config["lora"],
        "training": config["training"],
    }
    if saved != identity:
        raise ValueError(
            "Resume configuration differs from config.json at "
            + ", ".join(_configuration_differences(saved, identity)) + ". Use "
            "checkpoint_mode=weights to start a new optimizer."
        )
    if saved_model_config != model_config:
        raise ValueError(
            "Resume model settings differ from model.pt at "
            + ", ".join(_configuration_differences(saved_model_config, model_config, "config")) + ". Use "
            "checkpoint_mode=weights to start a new optimizer."
        )
    return config


def capture_rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
    }


def restore_rng_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def rng_state_path(checkpoint_dir, rank, world_size):
    name = "rng_state.pth" if world_size <= 1 else f"rng_state_{rank}.pth"
    return checkpoint_dir / name


def _validate_training_counters(state):
    for name in ("global_step", "data_pass", "batch_index"):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError(f"Trainer checkpoint has an invalid {name}")


def _read_trainer_state(path):
    """Read resumable metadata without copying memory-mapped optimizer tensors."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {"global_step", "data_pass", "batch_index", "world_size", "optimizer", "lr_scheduler"}
    if (not isinstance(state, dict) or set(state) != required
            or type(state["world_size"]) is not int or state["world_size"] <= 0
            or not isinstance(state["optimizer"], dict) or not isinstance(state["lr_scheduler"], dict)):
        raise ValueError(f"Invalid trainer checkpoint: {path}")
    _validate_training_counters(state)
    return state


def read_training_state(checkpoint):
    """Read resumable state on this process, or None for native base weights."""
    if checkpoint.is_dir():
        return None
    path = checkpoint.with_name(TRAINER_STATE_NAME)
    directory = checkpoint.parent
    if ((directory / CHECKPOINT_WRITING_NAME).exists() or
            ((directory / CHECKPOINT_COMPLETE_NAME).exists()
             and not checkpoint_is_complete(directory))):
        raise ValueError(f"Cannot resume an incomplete checkpoint: {directory}")
    return _read_trainer_state(path) if path.is_file() else None


def load_training_state(checkpoint, mode, rank, process_group=None):
    """Load the trainer sidecar, falling back to weights for base checkpoints."""
    if mode == "weights" or checkpoint.is_dir():
        return None
    state = distributed_call(
        lambda: read_training_state(checkpoint) if rank == 0 else None,
        process_group, f"Trainer checkpoint load {checkpoint.with_name(TRAINER_STATE_NAME)}",
    )
    payload = [None if state is None else {**state, "optimizer": None}]
    dist.broadcast_object_list(payload, src=0, group=process_group)
    return state if rank == 0 else payload[0]


def validate_training_state(
    state, identity, model_config, saved_model_config, world_size,
    max_train_steps, checkpoint, loader_info, *, data_steps=None,
    required_artifacts=None,
):
    if state["world_size"] != world_size:
        raise ValueError(
            f"Resume requires world_size={state['world_size']}, got "
            f"{world_size}. Use checkpoint_mode=weights to change GPU count."
        )
    validate_training_position(state, loader_info, data_steps=data_steps)
    if not 0 <= state["global_step"] <= max_train_steps:
        raise ValueError(
            f"Checkpoint step {state['global_step']} is outside "
            f"max_train_steps={max_train_steps}"
        )
    config = validate_checkpoint_identity(
        checkpoint, identity, model_config, saved_model_config,
    )
    if required_artifacts is None:
        required_artifacts = (ADAPTER_MODEL_NAME,) if identity["lora"] is not None else ()
    for name in required_artifacts:
        path = checkpoint.with_name(name)
        if not path.is_file():
            raise FileNotFoundError(path)
    return config






def retain_training_base(backend, model, base, model_config, output_dir, rank=0):
    """Preserve loaded dependencies before retention or same-step replacement."""
    if output_dir is None:
        return base
    output_dir = Path(output_dir).resolve()
    for source in (Path(base).absolute(), Path(base).resolve()):
        try:
            relative = source.relative_to(output_dir)
        except ValueError:
            continue
        if not relative.parts or not re.fullmatch(r"checkpoint-\d{6,}", relative.parts[0]):
            continue
        raw_state = model.state_dict()
        state = backend.normalize_state_dict(raw_state)
        retained = output_dir / "model_bases" / relative.parts[0] / "model.pt"
        if rank == 0:
            retained.parent.mkdir(parents=True, exist_ok=True)
            save_torch_state(retained, backend.checkpoint(state, config=model_config))
        return retained
    return base


def initialize_training_base(backend, model_dir, checkpoint, model_config, *,
                             lora_rank, saved_model=None, output_dir=None, rank=0):
    """Load and bind the basis before adapters or training precision can change it."""
    model = backend.build(model_dir, model_config)
    base = checkpoint if saved_model is None else Path(saved_model["base_checkpoint"])
    if saved_model is None or lora_rank:
        model = backend.load_initial_weights(model, base)
    else:
        backend.load_weights(model, checkpoint)
    if lora_rank:
        base = retain_training_base(backend, model, base, model_config, output_dir, rank)
    return model, base


def initialize_training_model(backend, model_dir, checkpoint, model_config,
                              training_state, *, lora_rank, lora_alpha,
                              lora_dropout, lora_targets, seed,
                              lora_checkpoint=None, eval_mode=False,
                              freeze_camera=False, output_dir=None, rank=0):
    """Load base or resumable weights, then apply the recipe's adapter policy."""
    torch.manual_seed(seed)
    saved_model = None if training_state is None else load_checkpoint_config(checkpoint)["model"]
    model, base = initialize_training_base(
        backend, model_dir, checkpoint, model_config, lora_rank=lora_rank,
        saved_model=saved_model, output_dir=output_dir, rank=rank,
    )
    resumed_adapter = training_state is not None and bool(lora_rank)
    model = backend.apply_lora(
        model, lora_rank, lora_alpha, lora_dropout, lora_targets, seed,
        freeze_camera=freeze_camera,
    )
    if resumed_adapter:
        backend.load_lora_weights(model, checkpoint.with_name(ADAPTER_MODEL_NAME))
    elif lora_checkpoint is not None:
        backend.load_lora_weights(model, lora_checkpoint)
    backend.enable_gradient_checkpointing(model)
    if eval_mode:
        model.eval()
    else:
        model.train()
    return model, base


def restore_training_state(
    model,
    optimizer,
    lr_scheduler,
    state,
    rank,
    device,
    checkpoint_dir,
    process_group=None,
):
    rng_path = rng_state_path(checkpoint_dir, rank, state["world_size"])
    def restore_rng():
        if not rng_path.is_file():
            raise FileNotFoundError(
                f"Missing RNG state: {rng_path}; use "
                "checkpoint_mode=weights to start a new optimizer"
            )
        rng_state = torch.load(
            rng_path, map_location="cpu", weights_only=False
        )
        restore_rng_state(rng_state, device)

    distributed_call(restore_rng, process_group, "Checkpoint RNG restore")
    full_optimizer = state["optimizer"] if rank == 0 else None
    distributed_call(
        lambda: restore_optimizer_state_dict(
            model,
            optimizer,
            full_optimizer,
            rank,
            checkpoint_dir,
        ), process_group, "Optimizer checkpoint scatter",
    )
    distributed_call(
        lambda: lr_scheduler.load_state_dict(state["lr_scheduler"]),
        process_group, "Optimizer checkpoint restore",
    )
    return (
        state["global_step"],
        state["data_pass"],
        state["batch_index"],
    )


def validate_training_position(state, info, *, data_steps=None):
    """Validate consumed data windows, including alternating optimizer updates."""
    _validate_training_counters(state)
    if data_steps is None:
        data_steps = state["global_step"]
    if type(data_steps) is not int or data_steps < 0:
        raise ValueError("Completed data steps must be a nonnegative integer")
    data_pass, step_in_pass = divmod(data_steps, info.steps_per_pass)
    batch_index = step_in_pass * (info.micro_batches // info.steps_per_pass)
    if (state["data_pass"], state["batch_index"]) != (data_pass, batch_index):
        raise ValueError(
            "Trainer data cursor does not match completed data steps: "
            f"expected pass={data_pass}, batch={batch_index}; "
            f"got pass={state['data_pass']}, batch={state['batch_index']}"
        )


def next_training_position(data_pass, batch_index, micro_batches):
    batch_index += 1
    if batch_index == micro_batches:
        return data_pass + 1, 0
    return data_pass, batch_index


def save_trainer_state(
    path,
    model,
    optimizer,
    lr_scheduler,
    rank,
    world_size,
    device,
    global_step,
    data_pass,
    batch_index,
    process_group=None,
):
    optimizer_state = distributed_call(
        lambda: full_optimizer_state_dict(model, optimizer),
        process_group, "Optimizer checkpoint gather",
    )

    def write_rng():
        save_torch_state(rng_state_path(path.parent, rank, world_size), capture_rng_state(device))

    def write_trainer():
        nonlocal optimizer_state
        # The unwound write frame owns the snapshot during error consensus.
        saved_optimizer, optimizer_state = optimizer_state, None
        if rank == 0:
            save_torch_state(path, {
                "global_step": global_step,
                "data_pass": data_pass,
                "batch_index": batch_index,
                "world_size": world_size,
                "optimizer": saved_optimizer,
                "lr_scheduler": lr_scheduler.state_dict(),
            })

    try:
        distributed_call(write_rng, process_group, "Checkpoint RNG write")
        distributed_call(write_trainer, process_group, "Trainer checkpoint write")
    finally:
        # RNG publication or a peer failure can stop before write_trainer runs.
        optimizer_state = None


def tracker_config(args, **extra):
    config = {
        name: asdict(value) if is_dataclass(value) else str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    config.update(extra)
    return config

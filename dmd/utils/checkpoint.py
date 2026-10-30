"""Complete DMD/AR training checkpoints alongside compatible generator exports."""

from pathlib import Path
from datetime import timedelta
import os
import random
import shutil
import socket
import tempfile

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig, FullStateDictConfig,
    FullyShardedDataParallel as FSDP, StateDictType,
)

from utils.distributed import EMA_FSDP, fsdp_state_dict
from utils.paths import save_config


def _collective_call(function, description, group=None):
    """Propagate local I/O/validation failures before the next collective."""
    result, error = None, None
    try:
        result = function()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError(f"{description}: " + "; ".join(
            f"rank {rank}: {item}" for rank, item in enumerate(errors) if item))
    return result


class _NodeCheckpointTransport:
    """Stream checkpoint files among node leaders over bounded CPU buffers."""
    CHUNK_BYTES = 16 * 1024 * 1024

    def __init__(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        local = (socket.gethostname(), int(os.environ.get('LOCAL_RANK', '0')))
        layout = [local]
        if dist.is_initialized():
            layout = [None] * self.world_size
            dist.all_gather_object(layout, local)
        by_host = {}
        for rank, (host, local_rank) in enumerate(layout):
            by_host.setdefault(host, []).append((rank, local_rank))
        leaders = {}
        for host, ranks in by_host.items():
            preferred = [rank for rank, local_rank in ranks if local_rank == 0]
            leaders[host] = min(preferred or [rank for rank, _ in ranks])
        self.owners = [leaders[host] for host, _ in layout]
        self.leaders = tuple(sorted(leaders.values()))
        if self.leaders[0] != 0:
            raise ValueError('Checkpoint transport requires global rank 0 to lead its node')
        self.group = None
        if len(self.leaders) > 1:
            # Every world rank creates groups in the same order; only the node
            # leaders participate in file-transfer collectives. NCCL never sees
            # the checkpoint payload or a whole optimizer dictionary.
            self.group = dist.new_group(ranks=list(self.leaders), backend='gloo',
                                        timeout=timedelta(minutes=30))

    def copy_file(self, directory, filename, source, destinations):
        destinations = tuple(rank for rank in destinations if rank != source)
        if not destinations or self.rank not in self.leaders:
            return
        if source not in self.leaders or any(rank not in self.leaders for rank in destinations):
            raise ValueError('Checkpoint transfer endpoints must be node leaders')
        path = Path(directory) / filename
        reader = writer = None
        try:
            size = _collective_call(
                lambda: path.stat().st_size if self.rank == source else None,
                f'Stat checkpoint {filename}', self.group)
            header = [size]
            dist.broadcast_object_list(header, src=source, group=self.group)
            size = header[0]

            def present():
                if self.rank not in destinations:
                    return True
                if not path.exists():
                    return False
                if not path.is_file() or path.stat().st_size != size:
                    raise ValueError(f'Conflicting checkpoint file: {path}')
                return True
            exists = _collective_call(present, f'Check checkpoint {filename}', self.group)
            copies = [None] * len(self.leaders)
            dist.all_gather_object(copies, exists, group=self.group)
            # A shared filesystem already exposes all files; retain support for
            # that layout without rewriting the source or publishing twice.
            if all(copies):
                return

            def open_file():
                nonlocal reader, writer
                if self.rank == source:
                    reader = path.open('rb')
                elif self.rank in destinations and not exists:
                    writer = path.open('xb')
            _collective_call(open_file, f'Open checkpoint {filename}', self.group)
            buffer = _collective_call(
                lambda: torch.empty(min(size, self.CHUNK_BYTES), dtype=torch.uint8, device='cpu'),
                f'Allocate checkpoint transfer buffer for {filename}', self.group)
            view = memoryview(buffer.numpy())
            for offset in range(0, size, self.CHUNK_BYTES):
                count = min(size - offset, self.CHUNK_BYTES)

                def read():
                    if reader is not None and reader.readinto(view[:count]) != count:
                        raise IOError(f'Short checkpoint read: {path} at {offset}')
                _collective_call(read, f'Read checkpoint {filename}', self.group)
                dist.broadcast(buffer[:count], src=source, group=self.group)

                def write():
                    if writer is not None and writer.write(view[:count]) != count:
                        raise IOError(f'Short checkpoint write: {path} at {offset}')
                _collective_call(write, f'Write checkpoint {filename}', self.group)

            def finish():
                if writer is not None:
                    writer.flush()
                    os.fsync(writer.fileno())
                if (self.rank == source or self.rank in destinations) and path.stat().st_size != size:
                    raise IOError(f'Incomplete transferred checkpoint: {path}')
            _collective_call(finish, f'Flush checkpoint {filename}', self.group)
        finally:
            if reader is not None:
                reader.close()
            if writer is not None:
                writer.close()


def _broadcast(value):
    payload = [value]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _training_config(config):
    ignored = {
        "generator_ckpt", "checkpoint_mode", "config_name", "logdir",
        "max_steps", "log_iters", "save_steps", "no_save", "disable_wandb", "training_eval",
    }
    return {key: value for key, value in OmegaConf.to_container(config, resolve=True).items()
            if key not in ignored and not key.startswith("wandb_")}


def _capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device).cpu() if torch.cuda.is_available() else None}


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _model_state(model):
    return fsdp_state_dict(model) if isinstance(model, FSDP) else _to_cpu(model.state_dict())


def _load_model(model, state):
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(
                                      offload_to_cpu=next(model.parameters()).device.type != "cpu",
                                      rank0_only=False)):
            model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)


def _optimizer_state(model, optimizer):
    if not isinstance(model, FSDP):
        return _to_cpu(optimizer.state_dict())
    offload = next(model.parameters()).device.type != "cpu"
    with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=offload, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=offload, rank0_only=True)):
        return FSDP.optim_state_dict(model, optimizer)


class TrainingCheckpoint:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._transport = None
        mode = config.get("checkpoint_mode", "resume")
        if mode not in ("resume", "weights"):
            raise ValueError("checkpoint_mode must be resume or weights")
        source = config.get("generator_ckpt")
        self.directory = Path(source).parent if source else None

        def read():
            if self.rank != 0 or mode == "weights" or self.directory is None:
                return None
            if self.directory.name.startswith(".checkpoint_model_"):
                raise ValueError("Cannot resume an unpublished checkpoint")
            sidecar = self.directory / "trainer.pt"
            if not sidecar.is_file():
                print("No trainer.pt found; loading generator weights only, starting at step 0.")
                return None
            if Path(source).name != "model.pt":
                raise ValueError("Resume requires checkpoint model.pt")
            state = torch.load(sidecar, map_location="cpu", weights_only=True, mmap=True)
            required = {"format", "step", "world_size", "config", "optimizers", "wandb"}
            names = {"generator", "critic"} if config.trainer == "score_distillation" else {"generator"}
            if (not isinstance(state, dict) or set(state) != required
                    or not isinstance(state["config"], dict)
                    or not isinstance(state["optimizers"], dict)
                    or set(state["optimizers"]) != names):
                raise ValueError("Invalid training checkpoint metadata or optimizer states")
            return state

        self.state = _collective_call(read, "Read training checkpoint")
        self.metadata = _broadcast(
            {key: value for key, value in self.state.items() if key != "optimizers"}
            if self.state is not None else None)
        self.resuming = self.metadata is not None
        if self.resuming:
            _collective_call(self._validate, "Validate training checkpoint")

    def _validate(self):
        state = self.metadata
        if state.get("format") != 1:
            raise ValueError("Unsupported training checkpoint format")
        if state["world_size"] != self.world_size:
            raise ValueError("Resume requires the same GPU count; use checkpoint_mode=weights to change it")
        if type(state["step"]) is not int or state["step"] < 0:
            raise ValueError("Invalid training step")
        maximum = self.config.get("max_steps")
        if maximum and int(maximum) < state["step"]:
            raise ValueError("max_steps is smaller than the checkpoint step")
        if self.config.seed == 0:
            self.config.seed = state["config"]["seed"]
        current = _training_config(self.config)
        saved = dict(state["config"])
        # Normalize the saved prefix-threshold key before comparing configurations.
        old_prefix_name = "history_guidance_min_history_frames"
        if old_prefix_name in saved:
            prefix = saved.pop(old_prefix_name)
            if "history_guidance_min_prefix_frames" in saved and saved["history_guidance_min_prefix_frames"] != prefix:
                raise ValueError("Conflicting saved supervision prefix thresholds")
            saved["history_guidance_min_prefix_frames"] = prefix
        differences = sorted(key for key in current.keys() | saved.keys()
                             if current.get(key) != saved.get(key))
        if differences:
            raise ValueError("Resume training settings differ: " + ", ".join(differences)
                             + "; use checkpoint_mode=weights for a new experiment")
        required = ["model.pt", "trainer.pt"]
        required += [f"rank_{rank:05d}.pt" for rank in range(self.world_size)]
        if self.config.trainer == "score_distillation":
            required.append("critic.pt")
        if self.rank == 0:
            for name in required:
                if not (self.directory / name).is_file():
                    raise FileNotFoundError(f"Incomplete training checkpoint: {self.directory / name}")
        if not self.config.get("disable_wandb", False):
            self.wandb_options()

    def wandb_options(self):
        run = self.metadata.get("wandb") if self.resuming else None
        if not run:
            return {}
        for key in ("entity", "project"):
            current = self.config.get("wandb_" + key)
            if current and current != run[key]:
                raise ValueError(f"Resume W&B {key} differs from the saved run")
        return {"id": run["id"], "resume": "allow"}

    @staticmethod
    def _models(trainer):
        result = {"generator": (trainer.model.generator, trainer.generator_optimizer)}
        if trainer.config.trainer == "score_distillation":
            result["critic"] = (trainer.model.fake_score, trainer.critic_optimizer)
        return result

    def restore(self, trainer):
        if not self.resuming:
            return
        if self._transport is None:
            self._transport = _NodeCheckpointTransport()
        transport = self._transport
        staged = None

        def prepare():
            nonlocal staged
            if self.rank == 0:
                return str(self.directory)
            if self.rank in transport.leaders:
                parent = Path(trainer.output_path)
                parent.mkdir(parents=True, exist_ok=True)
                staged = Path(tempfile.mkdtemp(prefix='.checkpoint_restore.', dir=parent))
                return str(staged)
        path = _collective_call(prepare, 'Prepare checkpoint restore staging')
        paths = [path]
        if dist.is_initialized():
            paths = [None] * self.world_size
            dist.all_gather_object(paths, path)
        directory = Path(paths[transport.owners[self.rank]])
        local_rng = None
        try:
            destinations = tuple(rank for rank in transport.leaders if rank != 0)
            files = ['model.pt']
            if trainer.config.trainer == 'score_distillation':
                files.append('critic.pt')
            # FSDP scatters optimizer state directly from global rank 0. Plain
            # distributed models also need a transient local optimizer sidecar.
            if any(not isinstance(model, FSDP) for model, _ in self._models(trainer).values()):
                files.append('trainer.pt')
            for filename in files:
                _collective_call(
                    lambda: transport.copy_file(directory, filename, 0, destinations),
                    f'Stage checkpoint {filename}')
            for rank, owner in enumerate(transport.owners):
                if owner != 0:
                    filename = f'rank_{rank:05d}.pt'
                    _collective_call(
                        lambda: transport.copy_file(directory, filename, 0, (owner,)),
                        f'Stage checkpoint {filename}')
            local_rng = self._restore(trainer, directory)
        finally:
            _collective_call(lambda: shutil.rmtree(staged) if staged is not None else None,
                             'Clean checkpoint restore staging')
        # File transport and data replay must not advance the next training
        # noise/timestep/rollout draws. Restore after all staging is removed.
        _restore_rng(local_rng, self.device)
        self.state = None
        if self.rank == 0:
            print(f"Resumed training at step {trainer.step} from {self.directory}")

    def _restore(self, trainer, directory):
        models = self._models(trainer)
        for name, (model, optimizer) in models.items():
            path = directory / ("model.pt" if name == "generator" else "critic.pt")
            state = _collective_call(
                lambda: torch.load(path, map_location="cpu", weights_only=True, mmap=True),
                f"Read {name} weights")
            if name == "generator":
                state = state["generator"]
            _collective_call(lambda: _load_model(model, state), f"Restore {name} weights")
            del state
            optimizer_state = self.state["optimizers"][name] if self.rank == 0 else None
            if isinstance(model, FSDP):
                optimizer_state = FSDP.scatter_full_optim_state_dict(
                    optimizer_state, model, optim=optimizer)
            elif self.rank != 0:
                # Unsharded CPU/distributed consumers can read their local copy;
                # do not broadcast a potentially large optimizer through NCCL.
                optimizer_state = _collective_call(
                    lambda: torch.load(directory / 'trainer.pt', map_location='cpu',
                                       weights_only=True, mmap=True)['optimizers'][name],
                    f'Read {name} optimizer')
            elif dist.is_initialized():
                # Match the error-propagation collective entered by peers.
                _collective_call(lambda: None, f'Read {name} optimizer')
            _collective_call(lambda: optimizer.load_state_dict(optimizer_state),
                             f"Restore {name} optimizer")

        def restore_local():
            local = torch.load(directory / f"rank_{self.rank:05d}.pt",
                               map_location="cpu", weights_only=False)
            steps = self.metadata["step"]
            data_steps = steps
            if trainer.config.trainer == "score_distillation":
                ratio = int(trainer.config.dfake_gen_update_ratio)
                data_steps += (steps + ratio - 1) // ratio
            consumed = data_steps * int(trainer.config.get("gradient_accumulate", 1) or 1)
            expected = divmod(consumed, len(trainer.dataloader.loader))
            if (local["data"]["epoch"], local["data"]["batch_index"]) != expected:
                raise ValueError("Checkpoint data position disagrees with completed optimizer steps")
            trainer.dataloader.load_state_dict(local["data"])
            ema = local["ema"]
            trainer.generator_ema = None
            if ema is not None:
                trainer.generator_ema = EMA_FSDP(trainer.model.generator, decay=ema["decay"])
                shadow = trainer.generator_ema.state_dict()
                if (shadow.keys() != ema["shadow"].keys() or any(
                        shadow[key].shape != value.shape for key, value in ema["shadow"].items())):
                    raise ValueError("Checkpoint EMA shards differ from the current FSDP layout")
                trainer.generator_ema.load_state_dict(ema["shadow"])
            trainer.step = steps
            return local["rng"]

        return _collective_call(restore_local, "Restore rank training state")

    def save(self, trainer):
        rng = _capture_rng(self.device)
        try:
            self._save(trainer, rng)
        finally:
            _restore_rng(rng, self.device)

    def _save(self, trainer, rng):
        target = Path(trainer.output_path) / f"checkpoint_model_{trainer.step:06d}"
        if self._transport is None:
            self._transport = _NodeCheckpointTransport()
        transport = self._transport
        temporary = None
        created_here = False

        def prepare():
            nonlocal temporary, created_here
            if self.rank in transport.leaders:
                target.parent.mkdir(parents=True, exist_ok=True)
            if self.rank == 0:
                if target.exists():
                    raise FileExistsError(f"Refusing to replace checkpoint: {target}")
                temporary = Path(tempfile.mkdtemp(prefix="." + target.name + ".", dir=target.parent))
                created_here = True
                return str(temporary)

        def prepare_local():
            nonlocal created_here
            if self.rank != 0 and self.rank in transport.leaders:
                try:
                    temporary.mkdir()
                    created_here = True
                except FileExistsError:
                    pass
        try:
            name = Path(_broadcast(_collective_call(prepare, "Prepare training checkpoint"))).name
            temporary = target.parent / name
            _collective_call(prepare_local, "Prepare node checkpoint directory")
            self._write_checkpoint(trainer, rng, temporary, target, transport,
                                   publish_local_config=created_here)
        finally:
            # Only remove a directory created by this invocation. On shared
            # filesystems other node leaders see rank 0's directory instead.
            _collective_call(
                lambda: shutil.rmtree(temporary) if created_here and temporary.exists() else None,
                'Clean checkpoint save staging')

    def _write_checkpoint(self, trainer, rng, temporary, target, transport,
                          publish_local_config=False):
        models = self._models(trainer)
        filenames = []
        for name, (model, _) in models.items():
            state = _model_state(model)
            if name == "generator":
                state = {"generator": state}
                if trainer.generator_ema is not None:
                    state["generator_ema"] = trainer.generator_ema.full_state_dict(model)
            filename = "model.pt" if name == "generator" else "critic.pt"
            filenames.append((filename, 0))
            _collective_call(lambda: torch.save(state, temporary / filename) if self.rank == 0 else None,
                             f"Save {name} weights")
            del state

        optimizers = {name: _optimizer_state(model, optimizer)
                      for name, (model, optimizer) in models.items()}
        local = {"rng": rng, "data": trainer.dataloader.state_dict(),
                 "ema": None if trainer.generator_ema is None else {
                     "decay": trainer.generator_ema.decay,
                     "shadow": trainer.generator_ema.state_dict()}}
        _collective_call(lambda: torch.save(local, temporary / f"rank_{self.rank:05d}.pt"),
                         "Save rank training state")

        def write_trainer():
            if self.rank != 0:
                return
            run = None
            if not self.config.get("disable_wandb", False):
                import wandb
                run = wandb.run
            state = {"format": 1, "step": trainer.step, "world_size": self.world_size,
                     "config": _training_config(self.config), "optimizers": optimizers,
                     "wandb": None if run is None else {
                         "id": run.id, "entity": run.entity, "project": run.project}}
            torch.save(state, temporary / "trainer.pt")
            save_config(self.config, temporary / "config.yaml",
                        checkpoint=target / "model.pt")
        _collective_call(write_trainer, "Save training checkpoint metadata")
        optimizers.clear()
        filenames.append(('trainer.pt', 0))
        filenames.append(('config.yaml', 0))
        filenames += [(f'rank_{rank:05d}.pt', transport.owners[rank])
                      for rank in range(self.world_size)]
        for filename, source in filenames:
            if source != 0:
                _collective_call(lambda: transport.copy_file(temporary, filename, source, (0,)),
                                 f'Gather checkpoint {filename}')

        def validate_files():
            if self.rank != 0:
                return
            for filename, _ in filenames:
                path = temporary / filename
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(f'Incomplete node checkpoint: {path}')
        _collective_call(validate_files, 'Validate complete training checkpoint')

        def publish():
            if self.rank == 0:
                temporary.rename(target)
                print(f"Training checkpoint saved to {target}")
            elif self.rank in transport.leaders and publish_local_config:
                # An independent local staging directory proves this node does
                # not share rank 0's filesystem. Retain only the tiny config so
                # both nodes can use the same resume/evaluation config path.
                save_config(self.config, target / "config.yaml",
                            checkpoint=target / "model.pt")
        _collective_call(publish, "Publish training checkpoint")

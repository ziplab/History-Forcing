"""Shared video/latent datasets, LMDB storage and collation for every backend."""

import json
import os
from contextlib import ExitStack
from pathlib import Path

import lmdb
import numpy as np
import torch

from utils import (
    block_lengths, block_starts, image_pixels, image_transform,
    interpolate_camera_poses, read_video_frames, resize_pixels,
)

LATENT_FRAME_SHAPE = (16, 60, 104)
PROMPT_FIELD = "prompts"
TEACHER_PROMPT_FIELD = "teacher_prompts"
TEACHER_PROMPT_MASK_FIELD = "teacher_prompt_masks"
REVISIT_FIELD = "revisit_start_latent"
# Stored arrays are model-neutral: actions may have any feature width and
# reference slots carry encoded visual context. Models validate their dimensions.
OPTIONAL_ARRAYS = {
    "actions": ("actions", np.float32),
    "condition_actions": ("condition_actions", np.float32),
    "reference_latents": ("reference_latent", np.float16),
    "reference_masks": ("reference_mask", np.uint8),
    "condition_pixels": ("condition_pixels", np.uint8),
    "image_condition_latents": ("image_condition_latent", np.float16),
    "image_embeddings": ("image_embedding", np.float16),
}


def _validate_array_fields(arrays, target_frames, condition_frames, latent_shape, index):
    """Validate model-neutral timelines at both LMDB storage boundaries."""
    camera_shapes = {
        'condition_viewmats': (condition_frames, 4, 4),
        'condition_intrinsics': (condition_frames, 3, 3),
        'viewmats': (target_frames, 4, 4),
        'intrinsics': (target_frames, 3, 3),
    }
    cameras = arrays.keys() & camera_shapes.keys()
    if cameras:
        if cameras != camera_shapes.keys():
            raise ValueError('Camera data must include target and condition viewmats and intrinsics')
        for name, shape in camera_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(f'Invalid {name} row {index}: {arrays[name].shape}')
    if 'condition_actions' in arrays and 'actions' not in arrays:
        raise ValueError('Condition actions require target actions')
    if 'actions' in arrays:
        actions = arrays['actions']
        if actions.ndim != 2 or actions.shape[0] != target_frames or actions.shape[1] <= 0:
            raise ValueError(f'Actions must follow the target timeline, row {index}')
        if 'condition_actions' not in arrays:
            raise ValueError('Action data must include its condition timeline')
        if arrays['condition_actions'].shape != (condition_frames, actions.shape[1]):
            raise ValueError(f'Actions must follow the condition timeline, row {index}')
    if ('reference_latent' in arrays) != ('reference_mask' in arrays):
        raise ValueError('Reference latents and masks must be supplied together')
    if 'reference_latent' in arrays:
        refs, mask = arrays['reference_latent'], arrays['reference_mask']
        if refs.ndim != 5 or refs.shape[1] != latent_shape[0] or mask.shape != refs.shape[:1] or (mask > 1).any():
            raise ValueError(f'Invalid reference slots or mask, row {index}')
    if 'condition_pixels' in arrays:
        pixels = arrays['condition_pixels']
        if condition_frames != 1 or pixels.ndim != 3 or pixels.shape[0] != 3:
            raise ValueError('Condition pixels require a single image prefix and uint8 [3,H,W]')
    if 'image_condition_latent' in arrays:
        features = arrays['image_condition_latent']
        if (features.ndim != 4 or features.shape[0] != condition_frames + target_frames
                or features.shape[1] <= 0 or tuple(features.shape[2:]) != latent_shape[1:]):
            raise ValueError('Image condition latents must cover the full prefix+target spatial timeline')
    if 'image_embedding' in arrays:
        embedding = arrays['image_embedding']
        if embedding.ndim != 2 or min(embedding.shape) <= 0:
            raise ValueError('Image embedding must have shape [tokens,channels]')


def validate_revisit_start_latent(value, total_frames=None, block_size=None):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{REVISIT_FIELD} must be positive, got {value!r}")
    if total_frames is not None and value >= total_frames:
        raise ValueError(
            f"{REVISIT_FIELD}={value} must leave at least one frame in "
            f"a {total_frames}-frame target"
        )
    if block_size is not None:
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if total_frames is None:
            valid_boundary = value >= block_size and value % block_size == 0
        else:
            lengths = block_lengths(total_frames, block_size)
            starts, _ = block_starts(lengths)
            valid_boundary = value in starts[1:]
        if not valid_boundary:
            raise ValueError(
                f"{REVISIT_FIELD} must be an interior AR block boundary, "
                f"got {value!r}"
            )
    return value


def read_shape(env, name):
    with env.begin() as transaction:
        value = transaction.get(f"{name}_shape".encode())
    if value is None:
        raise KeyError(f"LMDB is missing metadata key: {name}_shape")
    return tuple(map(int, value.decode().split()))


def read_row(env, name, index):
    with env.begin() as transaction:
        value = transaction.get(f"{name}_{index}_data".encode())
    if value is None:
        raise KeyError(f"LMDB is missing row: {name}_{index}_data")
    return value


def read_row_shape(env, name, index):
    with env.begin() as transaction:
        value = transaction.get(f"{name}_{index}_shape".encode())
    if value is None:
        raise KeyError(f"LMDB is missing row shape: {name}_{index}_shape")
    return tuple(map(int, value.decode().split()))


class _LMDBDataset:
    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def _get_env(self):
        pid = os.getpid()
        if self._env is not None and self._pid != pid:
            self.close()
        if self._env is None:
            self._env = lmdb.open(
                self.path, readonly=True, lock=False, readahead=False,
                meminit=False, max_readers=512,
            )
            self._pid = pid
        return self._env

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_pid"] = None
        return state

    def get_prompt(self, index):
        return read_row(
            self._get_env(), PROMPT_FIELD, index
        ).decode("utf-8")

    def get_prompts(self, index):
        """Return every text prompt needed to train one sample."""
        return (self.get_prompt(index),)

    def checkpoint_identity(self):
        """Identify an immutable training source without scanning tensor rows.

        File metadata detects replacement; the transaction ID also detects
        committed edits when a tool preserves the file's modification time.
        The source path and transaction describe the dataset version.
        """
        path = Path(self.path).resolve()
        with self._get_env().begin() as transaction:
            stat = (path / "data.mdb").stat()
            return {"path": str(path), "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns, "transaction_id": transaction.id()}

    def validate_packing_identity(self, identity):
        """Reject a stale or unverifiable pack without modifying its rows."""
        with self._get_env().begin() as transaction:
            stored = transaction.get(b'__packing_identity__')
        if stored != json.dumps(identity, sort_keys=True).encode():
            raise ValueError(f"Packed data at {self.path} differs from the source or model assets; repack first")

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._pid = None


class LatentSequenceDataset(_LMDBDataset):
    """Read raw prompts and continuous latent sequences from one LMDB.

    Variable-length LMDBs store ``latents_shape == [N]`` and an exact
    ``<field>_<index>_shape`` beside every binary row. Fixed-length
    ``[N,F,C,H,W]`` records are also accepted. The only public field names
    are ``prompts``, ``latents``, ``viewmats``, ``intrinsics`` and their
    ``condition_*`` counterparts. Optional actions use the same target/prefix
    split; reference_latents/reference_masks are persistent visual context.
    Latent channels and spatial size come from row metadata, not a model name.
    OPSD data may additionally carry one
    ``teacher_prompts`` value per row. SFT data may additionally pair it with
    a boolean ``teacher_prompt_masks`` target timeline; true frames use the
    teacher prompt and false frames use the primary prompt. A visual condition
    inherits the first target frame's prompt route. An all-false row remains a
    single-prompt sample, which permits both schemas in one batch.
    Camera interpretation belongs to the selected backend; this reader only validates
    and transfers arrays.
    """

    def __init__(self, path, load_latent=False):
        path = Path(path)
        if not (path / "data.mdb").is_file():
            raise FileNotFoundError(f"LMDB missing data.mdb: {path}")
        self.path = str(path)
        self.load_latent = load_latent
        with lmdb.open(
            self.path, readonly=True, lock=False, readahead=False,
            meminit=False, max_readers=512,
        ) as env:
            latent_shape = read_shape(env, "latents")
            prompt_shape = read_shape(env, PROMPT_FIELD)
            if len(latent_shape) == 5:
                self._fixed_frames = latent_shape[1]
                self._count = latent_shape[0]
            elif len(latent_shape) == 1:
                self._count = latent_shape[0]
                self._fixed_frames = None
            else:
                raise ValueError(
                    "Packed targets must be continuous [N,F,C,H,W] records; "
                    f"multi-axis shape {latent_shape} is unsupported"
                )
            if prompt_shape != (self._count,):
                raise ValueError(f"Unexpected prompts shape {prompt_shape}")
            if self._count <= 0:
                raise ValueError("A latent dataset must contain at least one row")
            row_shapes = (
                (latent_shape[1:],) * self._count if self._fixed_frames is not None
                else tuple(read_row_shape(env, "latents", index) for index in range(self._count))
            )
            if any(len(shape) != 4 or min(shape) <= 0 for shape in row_shapes):
                raise ValueError("Latent rows must have positive [F,C,H,W] dimensions")
            self.latent_frame_shape = row_shapes[0][1:]
            if any(shape[1:] != self.latent_frame_shape for shape in row_shapes):
                raise ValueError("One LMDB must use one latent channel/spatial shape")
            with env.begin() as transaction:
                shapes = {}
                for name in (
                    *OPTIONAL_ARRAYS, TEACHER_PROMPT_FIELD,
                    TEACHER_PROMPT_MASK_FIELD, "viewmats", "intrinsics",
                    "condition_latents", "condition_viewmats", "condition_intrinsics",
                    REVISIT_FIELD,
                ):
                    value = transaction.get(f"{name}_shape".encode())
                    shapes[name] = None if value is None else tuple(map(int, value.decode().split()))
                self.has_target = transaction.cursor().set_key(b"latents_0_data")
            self.optional_fields = tuple(name for name in OPTIONAL_ARRAYS if shapes[name] is not None)
            for name in self.optional_fields:
                if shapes[name] != (self._count,):
                    raise ValueError(f"Invalid {name} metadata")
            if ("reference_latents" in self.optional_fields) != ("reference_masks" in self.optional_fields):
                raise ValueError("Reference latents and masks must be supplied together")
            self.has_actions = "actions" in self.optional_fields

            teacher_prompt_shape = shapes[TEACHER_PROMPT_FIELD]
            if teacher_prompt_shape not in (None, (self._count,)):
                raise ValueError(
                    f"Unexpected {TEACHER_PROMPT_FIELD} shape"
                )
            self.has_teacher_prompt = teacher_prompt_shape is not None

            prompt_mask_shape = shapes[TEACHER_PROMPT_MASK_FIELD]
            if prompt_mask_shape is None:
                self.has_prompt_switch = False
            elif self.has_teacher_prompt and prompt_mask_shape == (self._count,):
                self.has_prompt_switch = True
            else:
                raise ValueError(
                    "Teacher prompt masks require teacher prompts and shape "
                    f"[{self._count}]"
                )

            target_camera = (shapes["viewmats"], shapes["intrinsics"])
            if any(value is not None for value in target_camera):
                expected = (
                    (
                        (self._count, self._fixed_frames, 4, 4),
                        (self._count, self._fixed_frames, 3, 3),
                    )
                    if self._fixed_frames is not None
                    else ((self._count,), (self._count,))
                )
                if target_camera != expected:
                    raise ValueError("Invalid camera metadata")
                self.has_camera = True
            else:
                self.has_camera = False

            condition_shapes = (
                shapes["condition_latents"],
                shapes["condition_viewmats"],
                shapes["condition_intrinsics"],
            )
            if condition_shapes[0] is not None:
                if (
                    self._fixed_frames is not None
                    or condition_shapes[0] != (self._count,)
                ):
                    raise ValueError("Invalid condition metadata")
                expected_cameras = (
                    ((self._count,), (self._count,))
                    if self.has_camera else (None, None)
                )
                if condition_shapes[1:] != expected_cameras:
                    raise ValueError("Invalid condition camera metadata")
                self.has_condition = True
            else:
                if condition_shapes[1:] != (None, None):
                    raise ValueError("Condition cameras require condition latents")
                self.has_condition = False

            revisit_shape = shapes[REVISIT_FIELD]
            if revisit_shape not in (None, (self._count,)):
                raise ValueError(f"Unexpected {REVISIT_FIELD} shape")
            self.has_revisit = revisit_shape is not None

            if "condition_actions" in self.optional_fields and not self.has_actions:
                raise ValueError("Condition actions require target actions")
            if self.has_actions and self.has_condition and "condition_actions" not in self.optional_fields:
                raise ValueError("Action data must include its condition timeline")

            if self._fixed_frames is not None:
                self.lengths = (self._fixed_frames,) * self._count
                self.condition_lengths = (0,) * self._count
            else:
                self.lengths = tuple(
                    row_shapes[index][0]
                    for index in range(self._count)
                )
                self.condition_lengths = (
                    tuple(
                        read_row_shape(env, "condition_latents", index)[0]
                        for index in range(self._count)
                    )
                    if self.has_condition else (0,) * self._count
                )
        self._env = None
        self._pid = None

    def __len__(self):
        return self._count

    def _prompt_mask(self, env, index):
        prompt_mask = self._array(
            env, TEACHER_PROMPT_MASK_FIELD, index, np.uint8
        )
        if prompt_mask.shape != (self.lengths[index],) or np.any(
            prompt_mask > 1
        ):
            raise ValueError(f"Invalid prompt mask row {index}")
        return prompt_mask

    def get_prompts(self, index):
        """Return distinct temporal prompt routes, excluding OPSD-only teacher text."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        primary = self.get_prompt(index)
        if not self.has_prompt_switch:
            return (primary,)
        env = self._get_env()
        prompt_mask = self._prompt_mask(env, index)
        if not bool(prompt_mask.any()):
            return (primary,)
        teacher = read_row(
            env, TEACHER_PROMPT_FIELD, index
        ).decode("utf-8")
        if not teacher:
            raise ValueError(f"Prompt-switch row {index} has an empty teacher prompt")
        return tuple(dict.fromkeys((primary, teacher)))

    @staticmethod
    def _array(env, name, index, dtype, shape=None):
        if shape is None:
            shape = read_row_shape(env, name, index)
        value = np.frombuffer(read_row(env, name, index), dtype=dtype).reshape(shape)
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite {name} row {index}")
        return value.copy()

    def _camera_item(self, env, index):
        if not self.has_camera:
            raise ValueError("Dataset has no camera metadata")
        target_frames = self.lengths[index]
        condition_frames = self.condition_lengths[index]
        if self._fixed_frames is not None:
            target_viewmats = self._array(env, "viewmats", index, np.float32, (target_frames, 4, 4))
            target_intrinsics = self._array(env, "intrinsics", index, np.float32, (target_frames, 3, 3))
            condition_viewmats = np.empty((0, 4, 4), dtype=np.float32)
            condition_intrinsics = np.empty((0, 3, 3), dtype=np.float32)
        else:
            target_viewmats = self._array(
                env, "viewmats", index, np.float32
            )
            target_intrinsics = self._array(
                env, "intrinsics", index, np.float32
            )
            condition_viewmats = (
                self._array(env, "condition_viewmats", index, np.float32)
                if self.has_condition
                else np.empty((0, 4, 4), dtype=np.float32)
            )
            condition_intrinsics = (
                self._array(env, "condition_intrinsics", index, np.float32)
                if self.has_condition
                else np.empty((0, 3, 3), dtype=np.float32)
            )
        result = {
            "viewmats": torch.from_numpy(target_viewmats),
            "intrinsics": torch.from_numpy(target_intrinsics),
            "condition_viewmats": torch.from_numpy(condition_viewmats),
            "condition_intrinsics": torch.from_numpy(condition_intrinsics),
        }
        _validate_array_fields(result, target_frames, condition_frames, self.latent_frame_shape, index)
        return result

    def camera_item(self, index):
        """Read one camera timeline without loading any video latent."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._camera_item(self._get_env(), index)

    def _sequence_item(self, env, index):
        condition_frames = self.condition_lengths[index]
        condition_latent = (
            self._array(env, "condition_latents", index, np.float16)
            if self.has_condition
            else np.empty((0, *self.latent_frame_shape), dtype=np.float16)
        )
        if condition_latent.shape != (condition_frames, *self.latent_frame_shape):
            raise ValueError(f"Invalid condition latent row {index}")
        result = {"condition_latent": torch.from_numpy(condition_latent)}
        if not self.has_camera:
            return result
        result.update(self._camera_item(env, index))
        return result

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        env = self._get_env()
        result = self._sequence_item(env, index)
        result["prompt"] = self.get_prompt(index)
        result["target_frames"] = self.lengths[index]
        result["latent_frame_shape"] = self.latent_frame_shape
        optional = {}
        for name in self.optional_fields:
            field, dtype = OPTIONAL_ARRAYS[name]
            array = self._array(env, name, index, dtype)
            optional[field] = torch.from_numpy(array)
        actions = optional.get('actions')
        if actions is not None and actions.ndim == 2 and 'condition_actions' not in optional:
            optional['condition_actions'] = actions.new_empty((0, actions.shape[1]))
        _validate_array_fields(optional, self.lengths[index], self.condition_lengths[index], self.latent_frame_shape, index)
        result.update(optional)
        if self.has_teacher_prompt:
            result["student_prompt"] = result["prompt"]
            result["teacher_prompt"] = read_row(
                env, TEACHER_PROMPT_FIELD, index
            ).decode("utf-8")
        if self.has_prompt_switch:
            prompt_mask = self._prompt_mask(env, index)
            if bool(prompt_mask.any()) and not result["teacher_prompt"]:
                raise ValueError(
                    f"Prompt-switch row {index} has an empty teacher prompt"
                )
            if bool(prompt_mask.any()):
                result["teacher_prompt_mask"] = torch.from_numpy(
                    prompt_mask.astype(np.bool_)
                )
        if self.has_revisit:
            value = np.frombuffer(
                read_row(env, REVISIT_FIELD, index), dtype=np.int32
            )
            if value.shape != (1,):
                raise ValueError(f"Invalid {REVISIT_FIELD} row {index}")
            result[REVISIT_FIELD] = int(value[0])
        if self.load_latent:
            if self._fixed_frames is not None:
                stored_shape = (self._fixed_frames, *self.latent_frame_shape)
                latent = self._array(env, "latents", index, np.float16, stored_shape)
            else:
                latent = self._array(env, "latents", index, np.float16)
            if latent.shape != (self.lengths[index], *self.latent_frame_shape):
                raise ValueError(f"Invalid target latent row {index}")
            result["clean_latent"] = torch.from_numpy(latent)
        return result


class OPSDLatentDataset:
    """Expose OPSD samples with explicit revisit metadata."""

    def __init__(self, path, temporal_layout):
        self.temporal_layout = temporal_layout
        self.dataset = LatentSequenceDataset(path, load_latent=False)
        condition_lengths = set(self.dataset.condition_lengths)
        try:
            for frames in condition_lengths:
                temporal_layout.prefix_blocks(frames)
            if not self.dataset.has_revisit:
                raise KeyError(f"OPSD data is missing the explicit {REVISIT_FIELD} field")
        except Exception:
            self.dataset.close()
            raise

    def __len__(self):
        return len(self.dataset)

    def get_prompt(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.dataset.get_prompt(index)

    def get_prompts(self, index):
        return self.dataset.get_prompts(index)

    def close(self):
        self.dataset.close()

    def checkpoint_identity(self):
        return self.dataset.checkpoint_identity()

    def _target_frames(self, index, raw_revisit):
        """Validate the stored revisit against the backend's complete AR timeline."""
        stored_frames = self.dataset.lengths[index]
        condition_frames = self.dataset.condition_lengths[index]
        validate_revisit_start_latent(raw_revisit, stored_frames)
        target_frames = self.temporal_layout.complete_target_frames(stored_frames, prefix_frames=condition_frames)
        blocks = self.temporal_layout.target_blocks(target_frames, prefix_frames=condition_frames)
        if len(blocks) < 2:
            raise ValueError('OPSD requires at least two complete target AR blocks')
        validate_revisit_start_latent(raw_revisit, target_frames)
        starts, _ = block_starts(blocks)
        offset = self.temporal_layout.generated_prefix_frames(condition_frames)
        if raw_revisit + offset not in starts[1:]:
            raise ValueError(f'{REVISIT_FIELD} must be an interior AR block boundary')
        return target_frames

    def metadata(self, index):
        """Inspect the rollout layout and prompts without loading visual tensors."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        env = self.dataset._get_env()
        value = np.frombuffer(read_row(env, REVISIT_FIELD, index), dtype=np.int32)
        if value.shape != (1,):
            raise ValueError(f'Invalid {REVISIT_FIELD} row {index}')
        raw_revisit = int(value[0])
        target_frames = self._target_frames(index, raw_revisit)
        result = dict(prompt=self.get_prompt(index), target_frames=target_frames,
                      condition_frames=self.dataset.condition_lengths[index], revisit_start_latent=raw_revisit)
        if self.dataset.has_teacher_prompt:
            result.update(student_prompt=result['prompt'],
                          teacher_prompt=read_row(env, TEACHER_PROMPT_FIELD, index).decode('utf-8'))
        return result

    def __getitem__(self, index):
        sample = self.dataset[index]
        target_frames = self._target_frames(index, sample[REVISIT_FIELD])
        sample.update(target_frames=target_frames, condition_frames=self.dataset.condition_lengths[index])
        for name in ("viewmats", "intrinsics", "actions"):
            if name in sample:
                sample[name] = sample[name][:target_frames]
        return sample


def _pad_sequence(value, length, *, repeat_last=False):
    if value.shape[0] == length:
        return value
    if value.shape[0] > length:
        raise ValueError("Cannot pad a sequence to a shorter length")
    if repeat_last and value.shape[0]:
        padding = value[-1:].expand(length - value.shape[0], *value.shape[1:])
    else:
        padding = value.new_zeros((length - value.shape[0], *value.shape[1:]))
    return torch.cat((value, padding), dim=0)


def _pad_camera(value, length):
    if value.shape[0] or length == 0:
        return _pad_sequence(value, length, repeat_last=True)
    identity = torch.eye(
        value.shape[-1], dtype=value.dtype, device=value.device
    )
    return identity[None].expand(length, -1, -1).clone()


def collate_latent_sequences(samples):
    """Pad continuous sequences and expose masks for model-native handling."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    loaded = ["clean_latent" in sample for sample in samples]
    if not all(loaded):
        raise ValueError("Training batches require target latents for every row")
    lengths = torch.tensor(
        [sample["clean_latent"].shape[0] for sample in samples], dtype=torch.long
    )
    condition_lengths = torch.tensor(
        [sample["condition_latent"].shape[0] for sample in samples],
        dtype=torch.long,
    )
    target_frames = int(lengths.max())
    condition_frames = int(condition_lengths.max())
    batch = {
        "prompt": [sample["prompt"] for sample in samples],
        "valid_frame_mask": torch.arange(target_frames)[None] < lengths[:, None],
        "condition_mask": (
            torch.arange(condition_frames)[None] < condition_lengths[:, None]
        ),
        "condition_latent": torch.stack([
            _pad_sequence(sample["condition_latent"], condition_frames)
            for sample in samples
        ]),
        "clean_latent": torch.stack([
            _pad_sequence(sample["clean_latent"], target_frames)
            for sample in samples
        ]),
    }
    switched = ["teacher_prompt_mask" in sample for sample in samples]
    if any(switched):
        batch["teacher_prompt"] = [
            (
                sample["teacher_prompt"]
                if uses_teacher else sample["prompt"]
            )
            for sample, uses_teacher in zip(samples, switched, strict=True)
        ]
        batch["teacher_prompt_mask"] = torch.stack([
            _pad_sequence(
                (
                    sample["teacher_prompt_mask"]
                    if uses_teacher else torch.zeros(
                        sample["clean_latent"].shape[0], dtype=torch.bool
                    )
                ),
                target_frames,
            )
            for sample, uses_teacher in zip(samples, switched, strict=True)
        ]) & batch["valid_frame_mask"]
    camera_rows = ["viewmats" in sample for sample in samples]
    if any(camera_rows) and not all(camera_rows):
        raise ValueError("A batch cannot mix rows with and without cameras")
    if all(camera_rows):
        batch.update(
            viewmats=torch.stack([
                _pad_camera(sample["viewmats"], target_frames)
                for sample in samples
            ]),
            intrinsics=torch.stack([
                _pad_camera(sample["intrinsics"], target_frames)
                for sample in samples
            ]),
            condition_viewmats=torch.stack([
                _pad_camera(sample["condition_viewmats"], condition_frames)
                for sample in samples
            ]),
            condition_intrinsics=torch.stack([
                _pad_camera(sample["condition_intrinsics"], condition_frames)
                for sample in samples
            ]),
        )
    for field, length in (("actions", target_frames), ("condition_actions", condition_frames)):
        present = [field in sample for sample in samples]
        if any(present):
            if not all(present):
                raise ValueError(f"A batch cannot mix rows with and without {field}")
            batch[field] = torch.stack([_pad_sequence(sample[field], length) for sample in samples])
    if any("image_condition_latent" in sample for sample in samples):
        if not all("image_condition_latent" in sample for sample in samples):
            raise ValueError("A batch cannot mix rows with and without image condition latents")
        full_frames = int((lengths + condition_lengths).max())
        batch["image_condition_latent"] = torch.stack([
            _pad_sequence(sample["image_condition_latent"], full_frames) for sample in samples
        ])
    for field in ("reference_latent", "reference_mask", "image_embedding"):
        present = [field in sample for sample in samples]
        if any(present):
            if not all(present):
                raise ValueError(f"Use zero masked slots for rows without {field}")
            batch[field] = torch.stack([sample[field] for sample in samples])
    revisit_rows = [REVISIT_FIELD in sample for sample in samples]
    if any(revisit_rows):
        if not all(revisit_rows):
            raise ValueError(
                f"A batch cannot mix rows with and without {REVISIT_FIELD}"
            )
        batch[REVISIT_FIELD] = torch.tensor([
            sample[REVISIT_FIELD] for sample in samples
        ], dtype=torch.long)
    return batch


def unbatch_latent_sequences(batch):
    """Return canonical sample views, removing only padding added by collate."""
    lengths = batch['valid_frame_mask'].sum(1).tolist()
    prefixes = batch['condition_mask'].sum(1).tolist()
    rows = []
    for i, (length, prefix) in enumerate(zip(lengths, prefixes, strict=True)):
        row = dict(prompt=batch['prompt'][i], target_frames=length,
                   latent_frame_shape=tuple(batch['clean_latent'].shape[2:]),
                   condition_latent=batch['condition_latent'][i, :prefix],
                   clean_latent=batch['clean_latent'][i, :length])
        for name in ('viewmats', 'intrinsics', 'actions', 'teacher_prompt_mask'):
            if name in batch:
                row[name] = batch[name][i, :length]
        for name in ('condition_viewmats', 'condition_intrinsics', 'condition_actions'):
            if name in batch:
                row[name] = batch[name][i, :prefix]
        if 'image_condition_latent' in batch:
            row['image_condition_latent'] = batch['image_condition_latent'][i, :prefix+length]
        for name in ('reference_latent', 'reference_mask', 'image_embedding', 'teacher_prompt', REVISIT_FIELD):
            if name in batch:
                value = batch[name][i]
                row[name] = int(value) if name == REVISIT_FIELD else value
        rows.append(row)
    return rows


def dataset_metadata(dataset):
    def describe(values):
        low, high = min(values), max(values)
        return low if low == high else f"{low}-{high}"
    return {"samples": len(dataset), "target_frames": describe(dataset.lengths),
            "condition_frames": describe(dataset.condition_lengths),
            "latent_shape": list(dataset.latent_frame_shape)}


def validate_dataset_path(path):
    if not (Path(path) / "data.mdb").is_file():
        raise FileNotFoundError(f"Expected the shared latent LMDB at {path}; pack raw media first")


def training_group_ids(dataset, backend, indices=None):
    """Group canonical length metadata without loading any sample tensors."""
    groups = {}
    result = []
    for index in range(len(dataset)) if indices is None else indices:
        key = backend.training_batch_key(dataset.condition_lengths[index], dataset.lengths[index])
        result.append(groups.setdefault(key, len(groups)))
    return result


def validate_dataset_fields(dataset, backend):
    """Check required packed-field metadata without loading sample tensors."""
    required = set(backend.required_data_fields)
    if backend.camera_control:
        required.update(("viewmats", "intrinsics"))
    for field in sorted(required):
        try:
            read_shape(dataset._get_env(), field)
        except KeyError as error:
            raise ValueError(f"{type(backend).__name__} requires packed field {field!r}") from error


def validate_training_dataset(dataset, backend):
    if dataset.latent_frame_shape != backend.latent_shape:
        raise ValueError(f"Packed latent shape {dataset.latent_frame_shape} does not match model {backend.latent_shape}")
    validate_dataset_fields(dataset, backend)
    groups = {}
    for group, prefix, length in zip(training_group_ids(dataset, backend),
                                     dataset.condition_lengths, dataset.lengths, strict=True):
        prefixes, lengths = groups.setdefault(group, ([], []))
        prefixes.append(prefix)
        lengths.append(length)
    for prefixes, lengths in groups.values():
        backend.validate_training_lengths(prefixes, lengths)


def read_manifest(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    rows = json.loads(text) if path.suffix == '.json' else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Manifest must contain a nonempty list of sample dictionaries")
    return rows


def _native_packing_options(options, actions, source, backend):
    """Keep block identities and the stored supervision boundary across backends."""
    options = dict(options)
    for field in ('condition_frames', 'num_latent_frames'):
        if field not in options:
            raise ValueError(f"Missing stored packing option: {field}")
    if isinstance(actions, str) or not actions or any(
            action not in source.interactive_spec().actions for action in actions):
        raise ValueError("Native actions must contain supported block names")
    condition = options['condition_frames']
    original = source.temporal_layout
    prefix = original.generated_prefix_frames(condition)
    target = len(actions) * original.block_size - prefix
    if options['num_latent_frames'] != condition + target:
        raise ValueError("Stored frame count differs from the native action timeline")
    revisit = options.get(REVISIT_FIELD)
    if revisit is not None:
        validate_revisit_start_latent(revisit, target)
        block, remainder = divmod(revisit + prefix, original.block_size)
        if remainder:
            raise ValueError("Stored supervision start must be a native target-block boundary")
    if source is not backend:
        native = backend.temporal_layout
        prefix = native.generated_prefix_frames(condition)
        options['num_latent_frames'] = condition + len(actions) * native.block_size - prefix
        if revisit is not None:
            options[REVISIT_FIELD] = block * native.block_size - prefix
            validate_revisit_start_latent(options[REVISIT_FIELD], options['num_latent_frames'] - condition)
        # Image preprocessing and resolution belong to the destination model.
        for field in ('size', 'resize_mode', 'resize_resample'):
            options.pop(field, None)
        options.update({key: value for key, value in backend.interactive_input_options().items()
                        if key in ('size', 'resize_mode', 'resize_resample')})
    if options.get(REVISIT_FIELD) is None:
        options.pop(REVISIT_FIELD, None)
    return options


def _stored_camera(path, frames):
    """Read the dataset camera without reconstructing it from action labels."""
    with np.load(path, allow_pickle=False) as saved:
        controls = {key: saved[key].copy() for key in ('viewmats', 'intrinsics', 'rgb_frame_indices')}
    views, intrinsics, indices = (controls[key] for key in ('viewmats', 'intrinsics', 'rgb_frame_indices'))
    if (views.shape != (frames, 4, 4) or intrinsics.shape != (frames, 3, 3)
            or indices.shape != (frames,) or not all(np.isfinite(value).all() for value in controls.values())
            or indices[0] < 0 or (np.diff(indices) <= 0).any()
            or not np.allclose(views[:, 3], [0, 0, 0, 1])
            or not np.allclose(intrinsics[:, 2], [0, 0, 1])
            or (intrinsics[:, (0, 1), (0, 1)] <= 0).any()):
        raise ValueError(f'Invalid stored camera timeline: {path}')
    return controls


def read_packing_source(path, backend, output):
    """Adapt Parquet image/control records to the existing file-based manifest.

    Stored cameras are used directly with their source backend. Other backends
    keep their native action conversion. Stored supervision boundaries
    follow the same action block when its native latent length changes. Backends
    without control use the images and timeline only. Pack ranks share the cache.
    """
    # Format and relative assets follow the selected filename, not its blob.
    path = Path(path).absolute()
    if path.suffix != '.parquet':
        return read_manifest(path), path.parent
    from filelock import FileLock
    from dataclasses import asdict
    from uuid import uuid4
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as parquet:
        layout = json.loads((parquet.schema_arrow.metadata or {}).get(b'world_model', b'null'))
        if not isinstance(layout, dict) or layout.get('version') != 1:
            raise ValueError("Parquet needs world_model metadata with per-trajectory packing options")
        model = f'{type(backend).__module__}:{type(backend).__qualname__}'
        from backbones import load_backend
        source = backend if layout.get('backend') == model else load_backend(layout['backend'])
        if parquet.schema_arrow.names != ['image', 'prompt', 'actions', 'source']:
            raise ValueError("Expected image, prompt, actions and source Parquet columns")
        records, entries = parquet.read(columns=['prompt', 'actions']).to_pylist(), layout['rows']
        if not entries or [entry['row_index'] for entry in entries] != list(range(len(entries))):
            raise ValueError("Parquet packing rows must preserve a complete ordered manifest")
        expected = {(i, j) for i, record in enumerate(records) for j in range(len(record['actions']))}
        pairs = [(entry['image_index'], entry['action_index']) for entry in entries]
        if len(set(pairs)) != len(pairs) or set(pairs) != expected:
            raise ValueError("Parquet packing metadata must cover every image/action pair exactly once")
        mode = backend.interactive_spec().control_mode
        rows, control_batches, cameras = [], [], []
        for entry in entries:
            image_index, action_index = entry['image_index'], entry['action_index']
            record = records[image_index]
            actions = record['actions'][action_index]
            options = _native_packing_options(entry['options'], actions, source, backend)
            prompt = backend.interactive_prompt(record['prompt'])
            row = {**options, 'image': f'images/{image_index:06}.image',
                   'prompt': prompt, 'teacher_prompt': prompt}
            if mode != 'none':
                row['camera'] = f'controls/{image_index:06}-{action_index}.npz'
                if mode == 'discrete':
                    row['actions'] = f'controls/{image_index:06}-{action_index}.npy'
            camera = entry.get('camera') if source is backend and mode == 'camera' else None
            if camera is not None:
                camera = Path(camera)
                if camera.is_absolute() or '..' in camera.parts:
                    raise ValueError('Stored camera path must stay inside the dataset directory')
                controls = _stored_camera(path.parent / camera, row['num_latent_frames'])
            else:
                controls = (backend.interactive_trajectory(actions, condition_frames=row['condition_frames'])
                            if mode != 'none' else {})
            control_frames = len(controls.get('viewmats', controls.get('actions', []))) if controls else row['num_latent_frames']
            if control_frames != row['num_latent_frames']:
                raise ValueError("Stored frame count differs from the native action timeline")
            rows.append(row)
            control_batches.append(controls)
            if camera is not None:
                source_camera = path.parent / camera
                stat = source_camera.stat()
                cameras.append([str(source_camera), stat.st_size, stat.st_mtime_ns])
        stat = path.stat()
        settings = {'source': [str(path), stat.st_size, stat.st_mtime_ns], 'backend': model,
                    'player': {name: getattr(backend, name) for name in dir(backend) if name.startswith('player_')},
                    'input': backend.interactive_input_options(), 'interactive': asdict(backend.interactive_spec()),
                    'temporal': asdict(backend.temporal_layout), 'cameras': cameras}
        settings = json.loads(json.dumps(settings))
        output = Path(output).resolve()
        index = output.parent / f'.{output.name}.source-{type(backend).__name__}.json'
        index.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(index) + '.lock'):
            if index.is_file():
                cached = json.loads(index.read_text(encoding='utf-8'))
                root = Path(cached['directory'])
                if cached.get('settings') == settings and (root / 'manifest.json').is_file():
                    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
                    if manifest['rows'] == rows and all((root / name).is_file() for name in manifest['files']):
                        return manifest['rows'], root
            root = index.parent / f'.{output.name}.source-{type(backend).__name__}-{uuid4().hex[:12]}'
            (root / 'images').mkdir(parents=True)
            (root / 'controls').mkdir()
            images = parquet.read(columns=['image'])['image']
            files = set()
            for row, entry, controls in zip(rows, entries, control_batches, strict=True):
                image_name = row['image']
                if image_name not in files:
                    payload = (images[entry['image_index']].as_py() or {}).get('bytes')
                    if not isinstance(payload, bytes) or not payload:
                        raise ValueError('Parquet image bytes must be embedded')
                    (root / image_name).write_bytes(payload)
                    files.add(image_name)
                if 'viewmats' in controls:
                    np.savez(root / row['camera'], **{key: controls[key] for key in (
                        'viewmats', 'intrinsics', 'rgb_frame_indices')})
                    files.add(row['camera'])
                if 'actions' in controls:
                    np.save(root / row['actions'], controls['actions'], allow_pickle=False)
                    files.add(row['actions'])
            (root / 'manifest.json').write_text(json.dumps({'rows': rows, 'files': sorted(files)}), encoding='utf-8')
            temporary = index.with_suffix('.tmp')
            temporary.write_text(json.dumps({'settings': settings, 'directory': str(root)}), encoding='utf-8')
            temporary.replace(index)
        return rows, root


def camera_timeline(path, positions, source_size, size, mode):
    """Sample absolute w2c / normalized K at the selected RGB frame positions."""
    with np.load(path, allow_pickle=False) as camera:
        viewmats = camera['viewmats'].astype(np.float64)
        intrinsics = camera['intrinsics'].astype(np.float64)
        indices = camera['rgb_frame_indices'].astype(np.float64)
    count = len(viewmats)
    if (viewmats.shape != (count, 4, 4) or intrinsics.shape != (count, 3, 3)
            or indices.shape != (count,) or not count
            or not all(np.isfinite(x).all() for x in (viewmats, intrinsics, indices))
            or (np.diff(indices) <= 0).any() or indices[0] < 0
            or not np.allclose(viewmats[:, 3], [0, 0, 0, 1])
            or not np.allclose(intrinsics[:, 2], [0, 0, 1])
            or (intrinsics[:, (0, 1), (0, 1)] <= 0).any()):
        raise ValueError(f"Invalid camera timeline: {path}")
    rotations = viewmats[:, :3, :3]
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4)):
        raise ValueError(f"Camera viewmats must contain proper rigid rotations: {path}")
    if count == 1:
        # A single calibrated view explicitly denotes a static camera.
        poses = torch.from_numpy(np.repeat(viewmats, len(positions), axis=0)).float()
        ks = np.repeat(intrinsics, len(positions), axis=0)
    else:
        if positions[0] < indices[0] or positions[-1] > indices[-1]:
            raise ValueError(f"Camera does not cover the selected video span: {path}")
        c2w = np.linalg.inv(viewmats)
        poses = torch.linalg.inv(interpolate_camera_poses(
            indices, c2w[:, :3, :3], c2w[:, :3, 3], positions))
        ks = np.stack([np.interp(positions, indices, column)
                       for column in intrinsics.reshape(count, 9).T], -1).reshape(-1, 3, 3)
    _, transform = image_transform(source_size, size, mode)
    source_h, source_w = source_size
    height, width = size
    transform = np.diag([1 / width, 1 / height, 1]) @ transform @ np.diag([source_w, source_h, 1])
    return poses, torch.from_numpy(transform @ ks).float()


class VideoDataset(torch.utils.data.Dataset):
    """One CPU input contract for all sources; source adapters only create rows.

    Decoding is lazy and leaves uint8 pixels on CPU. ``decoded_frames`` specifies
    the codec timeline, without importing any backbone into the dataset.
    """

    def __init__(self, rows, root, *, pixel_shape, decoded_frames):
        self.rows, self.root = list(rows), Path(root)
        self.pixel_shape, self.decoded_frames = tuple(pixel_shape), decoded_frames
        if not self.rows:
            raise ValueError("Cannot pack an empty dataset")
        sizes = set()
        for row in self.rows:
            frames, prefix = self.lengths(row)
            if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError("Every sample needs a nonempty prompt")
            if 'teacher_prompt_mask' in row and 'teacher_prompt' not in row:
                raise ValueError("teacher_prompt_mask requires teacher_prompt")
            if 'image' in row and 'video' in row:
                raise ValueError("Specify image or a full condition+target video, not both")
            if 'poses' in row or 'intrinsics' in row:
                raise ValueError("Use camera NPZ with viewmats, intrinsics and rgb_frame_indices")
            if ('image' in row and prefix != 1) or (prefix and not {'image', 'video'} & row.keys()):
                raise ValueError("An image requires one condition frame; a video supplies a longer prefix")
            size = tuple(row.get('size', self.pixel_shape))
            if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                raise ValueError("size must be positive integer [height,width]")
            image_transform(size, size, row.get('resize_mode', 'fit'))
            sizes.add(size)
            self.frame_indices(row, frames)
        if len(sizes) != 1:
            raise ValueError("One LMDB must use one output spatial size")
        self.keep_condition_pixels = all(self.lengths(row)[1] == 1 for row in self.rows)

    def requires_encoding(self, start=0):
        """Whether the remaining rows contain pixels for a frozen encoder."""
        return any({'image', 'video', 'references'} & row.keys() for row in self.rows[start:])

    @staticmethod
    def lengths(row):
        frames = row['num_latent_frames']
        prefix = row.get('condition_frames', 1 if 'image' in row or 'video' in row else 0)
        if type(frames) is not int or type(prefix) is not int or not 0 <= prefix < frames:
            raise ValueError("condition_frames must leave a nonempty target timeline")
        return frames, prefix

    def frame_indices(self, row, frames):
        count = self.decoded_frames(frames)
        start = row.get('start_frame', 0)
        if type(start) is not int or start < 0:
            raise ValueError("start_frame must be a nonnegative integer")
        if 'frame_indices' in row and 'start_frame' in row:
            raise ValueError("Specify frame_indices or start_frame")
        indices = np.asarray(row['frame_indices'] if 'frame_indices' in row else np.arange(start, start + count))
        if (indices.shape != (count,) or indices.dtype.kind not in 'iu' or indices[0] < 0
                or (indices[1:] <= indices[:-1]).any()):
            raise ValueError(f"frame_indices must contain {count} strictly increasing RGB indices")
        return indices

    def input_paths(self):
        """Files affecting encoded samples, for resumable packing identity."""
        paths = set()
        for row in self.rows:
            for field in ('image', 'video', 'camera', 'actions', 'teacher_prompt_mask'):
                if field in row:
                    paths.add((self.root / row[field]).resolve())
            paths.update((self.root / path).resolve() for path in row.get('references', []))
        return sorted(paths)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frames, prefix = self.lengths(row)
        size, mode = tuple(row.get('size', self.pixel_shape)), row.get('resize_mode', 'fit')
        indices = self.frame_indices(row, frames)
        # Frame 0 is special for causal codecs; the remaining knots use the
        # exact same RGB selection as decoding, including a nonzero clip start.
        latent_positions = indices[[self.decoded_frames(i + 1) - 1 for i in range(frames)]]
        source_size, pixels, original = size, None, None

        def transform(image):
            nonlocal source_size
            source_size = tuple(image.shape[-2:])
            return resize_pixels(image, size, mode, resample=row.get('resize_resample', 'lanczos'))

        if 'video' in row:
            pixels = read_video_frames(self.root / row['video'], frame_indices=indices, transform=transform)
            if pixels is None:
                raise ValueError(f"Incomplete RGB span for sample {index}: {row['video']}")
            pixels = pixels.permute(1, 0, 2, 3)
        elif 'image' in row:
            original = image_pixels(self.root / row['image'])
            pixels = transform(original)[:, None]
        sample = {'prompt': row['prompt'], 'num_latent_frames': frames, 'condition_frames': prefix,
                  'size': size, 'pixels': pixels, 'has_target': 'video' in row}
        if prefix == 1:
            sample['condition_pixels'] = original if original is not None else pixels[:, 0]
        if 'camera' in row:
            poses, ks = camera_timeline(self.root / row['camera'], latent_positions,
                                       source_size, size, mode)
            sample.update(viewmats=poses[prefix:], condition_viewmats=poses[:prefix],
                          intrinsics=ks[prefix:], condition_intrinsics=ks[:prefix])
        if 'actions' in row:
            path = self.root / row['actions']
            array = np.load(path, allow_pickle=False) if path.suffix == '.npy' else np.asarray(
                json.loads(path.read_text(encoding='utf-8')))
            if array.ndim != 2 or len(array) != frames or not np.isfinite(array).all():
                raise ValueError("Actions must be finite [full_latent_frames,features]")
            actions = torch.from_numpy(array.copy()).float()
            sample.update(actions=actions[prefix:], condition_actions=actions[:prefix])
        if 'references' in row:
            paths = row['references']
            if not isinstance(paths, list) or not paths:
                raise ValueError("References must be a nonempty ordered list of image paths")
            side = row.get('reference_size', 512)
            if type(side) is not int or side <= 0:
                raise ValueError("reference_size must be a positive integer")
            sample['reference_pixels'] = [resize_pixels(image_pixels(self.root / path), (side, side), 'resize')
                                          for path in paths]
        for field in ('revisit_start_latent', 'teacher_prompt'):
            if field in row:
                sample[field] = row[field]
        if 'teacher_prompt_mask' in row:
            mask = np.load(self.root / row['teacher_prompt_mask'], allow_pickle=False)
            if mask.shape != (frames - prefix,) or not np.isin(mask, [0, 1]).all():
                raise ValueError("teacher_prompt_mask must cover exactly the target timeline")
            sample['teacher_prompt_mask'] = torch.from_numpy(mask.astype(np.uint8))
        return sample


def packing_identity(dataset, backend, model_dir):
    """Shared source/encoder settings for packing and inference validation."""
    model_dir = Path(model_dir).resolve()
    paths = set(dataset.input_paths())
    if dataset.requires_encoding():
        paths.update(Path(path).resolve() for path in backend.packing_assets(model_dir))
    return {'schema': 5, 'backend': f'{type(backend).__module__}:{type(backend).__qualname__}',
            'root': str(dataset.root.resolve()), 'rows': dataset.rows, 'model_dir': str(model_dir),
            'pixel_shape': list(backend.pixel_shape), 'latent_shape': list(backend.latent_shape),
            'files': [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in sorted(paths)]}


def write_latent_lmdb(path, samples, *, identity=None):
    """Stream the shared sample contract into variable-length LMDB rows.

    Inference-only records omit clean_latent and provide target_frames and
    latent_frame_shape. Their target shape is stored, but no fake GT is written.
    Every sample is committed before requesting the next encoded video. Fresh
    packs accept an iterable or a factory receiving the committed row count.
    Resuming committed rows requires that factory and a matching identity,
    so encoding starts at the stored offset. Sources and their iterators close
    on completion or failure; only a completed database is published at path.
    """
    path = Path(path).resolve()
    signature = json.dumps(identity, sort_keys=True).encode() if identity is not None else None
    if (path / "data.mdb").exists():
        if signature is not None:
            with lmdb.open(str(path), readonly=True, lock=False) as existing:
                with existing.begin() as transaction:
                    if transaction.get(b'__packing_identity__') == signature:
                        print(f"Reusing {path}", flush=True)
                        return
        raise FileExistsError(f"Refusing to overwrite existing LMDB: {path}")
    temporary = path.with_name('.' + path.name + '.building')
    if (temporary / 'data.mdb').exists() and signature is None:
        raise FileExistsError(f"Interrupted pack requires a matching identity to resume: {temporary}")
    temporary.mkdir(parents=True, exist_ok=True)
    schema = None
    with ExitStack() as resources:
        env = lmdb.open(str(temporary), map_size=1 << 36)
        resources.callback(env.close)
        with env.begin(write=True) as transaction:
            previous = transaction.get(b'__packing_identity__')
            completed = int(transaction.get(b'__count__') or b'0')
            if previous != signature and (previous is not None or completed):
                raise ValueError(f"Interrupted pack uses different inputs or encoder: {temporary}")
            if signature is not None:
                transaction.put(b'__packing_identity__', signature)
            stored_schema = transaction.get(b'__schema__')
            if stored_schema is not None:
                schema = json.loads(stored_schema)
        print(f"Pack: {completed} rows committed -> {path}", flush=True)
        if completed and not callable(samples):
            raise ValueError(f"Resuming {completed} committed rows requires a sample factory "
                             "accepting the resume offset")
        source = samples(completed) if callable(samples) else samples
        close = getattr(source, 'close', None)
        if close is not None:
            resources.callback(close)
        iterator = iter(source)
        if iterator is not source and (close := getattr(iterator, 'close', None)) is not None:
            resources.callback(close)
        for index, sample in enumerate(iterator, completed):
            clean = sample.get("clean_latent")
            shape = (tuple(clean.shape) if clean is not None else
                     (sample["target_frames"], *sample["latent_frame_shape"]))
            if len(shape) != 4 or any(isinstance(axis, bool) or not isinstance(axis, (int, np.integer))
                                      or axis <= 0 for axis in shape):
                raise ValueError(f"Targets require a positive integer [F,C,H,W] shape, row {index}")
            shape = tuple(map(int, shape))
            condition = sample["condition_latent"]
            if condition.ndim != 4 or tuple(condition.shape[1:]) != shape[1:]:
                raise ValueError("Condition and target latents must share [C,H,W]")
            arrays = {"condition_latents": (condition, np.float16)}
            if clean is not None:
                arrays["latents"] = (clean, np.float16)
            for field in ("viewmats", "intrinsics", "condition_viewmats", "condition_intrinsics"):
                if field in sample:
                    arrays[field] = (sample[field], np.float32)
            for stored, (field, dtype) in OPTIONAL_ARRAYS.items():
                if field in sample:
                    arrays[stored] = (sample[field], dtype)
            texts = {PROMPT_FIELD: sample["prompt"]}
            if "teacher_prompt" in sample:
                texts[TEACHER_PROMPT_FIELD] = sample["teacher_prompt"]
            if "teacher_prompt_mask" in sample:
                if TEACHER_PROMPT_FIELD not in texts:
                    raise ValueError("teacher_prompt_mask requires teacher_prompt")
                arrays[TEACHER_PROMPT_MASK_FIELD] = (sample["teacher_prompt_mask"], np.uint8)
            if REVISIT_FIELD in sample:
                revisit = validate_revisit_start_latent(sample[REVISIT_FIELD], shape[0])
                arrays[REVISIT_FIELD] = (np.array([revisit]), np.int32)
            fields = set(arrays) | set(texts) | {"latents"}
            current_schema = {'fields': sorted(fields), 'shape': list(shape[1:]), 'has_target': clean is not None}
            if schema is not None and current_schema != schema:
                raise ValueError("One LMDB must share latent shape, target availability and optional fields")
            schema = current_schema
            values = {}
            for name, (value, dtype) in arrays.items():
                if torch.is_tensor(value):
                    value = value.detach().cpu()
                    if value.dtype == torch.bfloat16:
                        value = value.float()
                    value = value.numpy()
                value = np.asarray(value)
                if value.dtype.kind not in 'biuf':
                    raise ValueError(f"Packed field {name}, row {index} must contain real numeric values")
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite packed field {name}, row {index}")
                if np.issubdtype(dtype, np.integer):
                    bounds = np.iinfo(dtype)
                    maximum = 1 if name in (TEACHER_PROMPT_MASK_FIELD, 'reference_masks') else bounds.max
                    if ((value < bounds.min).any() or (value > maximum).any()
                            or (value.dtype.kind == 'f' and (value != np.floor(value)).any())):
                        raise ValueError(f"Packed field {name}, row {index} must contain integers "
                                         f"in [{bounds.min}, {maximum}]")
                value = np.ascontiguousarray(value, dtype=dtype)
                if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
                    raise ValueError(f"Non-finite packed field {name}, row {index}")
                if name == TEACHER_PROMPT_MASK_FIELD:
                    if value.shape != (shape[0],):
                        raise ValueError(f"Invalid prompt mask row {index}")
                    if value.any() and not texts[TEACHER_PROMPT_FIELD]:
                        raise ValueError(f"Prompt-switch row {index} has an empty teacher prompt")
                values[name] = value
            array_fields = {OPTIONAL_ARRAYS[name][0] if name in OPTIONAL_ARRAYS else name: value
                            for name, value in values.items()}
            _validate_array_fields(array_fields, shape[0], condition.shape[0], shape[1:], index)
            with env.begin(write=True) as transaction:
                transaction.put(b'__count__', str(index + 1).encode())
                transaction.put(b'__schema__', json.dumps(schema).encode())
                for name in fields:
                    transaction.put(f"{name}_shape".encode(), str(index + 1).encode())
                transaction.put(f"latents_{index}_shape".encode(), ' '.join(map(str, shape)).encode())
                for name, value in values.items():
                    transaction.put(f"{name}_{index}_data".encode(), value.tobytes())
                    transaction.put(f"{name}_{index}_shape".encode(), ' '.join(map(str, value.shape)).encode())
                for name, value in texts.items():
                    transaction.put(f"{name}_{index}_data".encode(), value.encode('utf-8'))
            print(f"Packed {index + 1} samples -> {path}", flush=True)
        if schema is None:
            raise ValueError("Cannot pack an empty dataset")
        env.sync()
    os.replace(temporary, path)

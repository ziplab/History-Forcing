"""Temporal inputs shared by Wan backbones and camera-conditioning mechanisms.

Camera data uses W2C matrices and normalized 3x3 intrinsics. Each mechanism
prepares its native representation before time slicing; model hooks install
and inject the corresponding attention or feature modules.
"""

from dataclasses import dataclass, field, fields, is_dataclass

from utils import block_lengths


def batch_inputs(values):
    """Join prepared tensor conditions on batch axis 0, preserving static metadata."""
    import torch
    if not values:
        raise ValueError('Cannot batch an empty condition list')
    first = values[0]
    if any(type(value) is not type(first) for value in values):
        raise TypeError('Condition types must agree within a batch')
    if len(values) == 1:
        return first
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        if any(value.keys() != first.keys() for value in values):
            raise ValueError('Condition fields must agree within a batch')
        return {key: batch_inputs([value[key] for value in values]) for key in first}
    if is_dataclass(first):
        return type(first)(**{item.name: batch_inputs([getattr(value, item.name) for value in values])
                              for item in fields(first)})
    if isinstance(first, (str, int, float, bool, type(None))) and all(value == first for value in values):
        return first
    raise TypeError('Conditions must contain tensors, dataclasses, mappings or equal scalar metadata')


@dataclass(frozen=True)
class TemporalInputs:
    """Named forward inputs, with time on axis 1 and separate static values.

    A temporal value may also be a backend-owned object implementing ``[:, t]``
    (for example a lazy camera-ray map). Slicing produces views;
    model projection and device/dtype placement remain the backend's work.
    """

    temporal: dict = field(default_factory=dict)
    static: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.temporal.keys() & self.static.keys():
            raise ValueError("An input cannot be both temporal and static")

    def slice(self, start, length):
        return type(self)({name: value[:, start:start + length] for name, value in self.temporal.items()},
                          self.static)

    def forward_kwargs(self):
        return {**self.static, **self.temporal}

    def pad(self, frames, *, repeat_last=True):
        """Pad masked SFT tails by repeating the last tensor or appending zeros.

        Static references are retained without padding or copying.
        Structured controls with a different padding policy are prepared by
        the backend's training_inputs hook instead.
        """
        import torch
        values = {}
        for name, value in self.temporal.items():
            if not torch.is_tensor(value) or value.ndim < 2:
                raise TypeError('Structured temporal controls require backend training_inputs')
            length = value.shape[1]
            if not 0 < length <= frames:
                raise ValueError('Temporal padding requires a nonempty input no longer than the target')
            if length == frames:
                values[name] = value
            else:
                tail = (value[:, -1:].expand(-1, frames-length, *value.shape[2:]) if repeat_last
                        else value.new_zeros((value.shape[0], frames-length, *value.shape[2:])))
                values[name] = torch.cat((value, tail), dim=1)
        return type(self)(values, self.static)


def fixed_camera_inputs(inputs, prefix_frames, target_frames, *, reference):
    """Hold the camera in prepared coordinates while preserving static inputs."""
    import torch

    if reference == 'previous_block':
        views = inputs.temporal['viewmats']
        return TemporalInputs({**inputs.temporal, 'viewmats':
                               torch.eye(4, device=views.device, dtype=views.dtype).expand_as(views).clone()},
                              inputs.static)
    values = {}
    anchor = max(prefix_frames - 1, 0)
    for name, value in inputs.temporal.items():
        fixed = value.clone()
        fixed[:, prefix_frames:prefix_frames + target_frames] = value[:, anchor:anchor + 1]
        values[name] = fixed
    return TemporalInputs(values, inputs.static)


@dataclass(frozen=True)
class CausalPrefix:
    """Clean video prefix and its projected tokens, with aligned native controls."""

    latent: object
    embedding: object
    inputs: TemporalInputs


@dataclass(frozen=True)
class RolloutConditioning:
    """Backend-prepared prefix, moving student and fixed teacher controls."""

    prefix: TemporalInputs
    student: TemporalInputs
    teacher: TemporalInputs


@dataclass(frozen=True)
class TemporalLayout:
    """Model policy, expressed in latent frames rather than attention tokens.

    A temporal image prefix can either roll with history or add a persistent
    sink outside the normal local window. Reference-image slots are separate
    model inputs and do not change this temporal layout.
    """

    block_size: int
    independent_first_frame: bool = False
    image_sink: bool = True
    prefill_condition: bool = True

    def __post_init__(self):
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("AR block size must be a positive integer")
        if not self.prefill_condition and (self.independent_first_frame or self.image_sink):
            raise ValueError("Generated image conditioning belongs to its full block, without an extra image sink")

    def prefix_blocks(self, frames):
        if type(frames) is not int or frames < 0:
            raise ValueError("Condition length must be a nonnegative integer")
        if frames == 0:
            return ()
        if not self.prefill_condition:
            if frames != 1:
                raise ValueError("Generated image conditioning requires exactly one prefix frame")
            return ()
        if frames == 1:
            return (1,)
        if self.independent_first_frame:
            return (1,) + block_lengths(frames - 1, self.block_size)
        return block_lengths(frames, self.block_size)

    def target_blocks(self, frames, *, prefix_frames):
        """Partition a target, including a generated first frame for native T2V."""
        frames += self.generated_prefix_frames(prefix_frames)
        if self.independent_first_frame and prefix_frames == 0:
            if frames == 1:
                return (1,)
            return (1,) + block_lengths(frames - 1, self.block_size)
        return block_lengths(frames, self.block_size)

    def complete_target_frames(self, frames, *, prefix_frames):
        """Drop only an incomplete trailing block from a source target."""
        if type(frames) is not int or frames <= 0:
            raise ValueError("Target length must be a positive integer")
        generated_prefix = self.generated_prefix_frames(prefix_frames)
        first = int(self.independent_first_frame and prefix_frames == 0)
        return first + (frames + generated_prefix - first) // self.block_size * self.block_size - generated_prefix

    def generated_prefix_frames(self, prefix_frames):
        """Condition positions generated inside the first block, before target index 0."""
        self.prefix_blocks(prefix_frames)
        return 0 if self.prefill_condition else prefix_frames

    def extra_sink_frames(self, prefix_frames):
        return int(self.prefill_condition and self.image_sink and prefix_frames == 1)

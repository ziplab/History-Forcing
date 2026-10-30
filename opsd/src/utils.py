# Localized from cvlab-kaist/WorldKV.
# Apache-2.0; see repository LICENSE.
"""Camera geometry, autoregressive block partitioning and RGB media loading."""

import os
import random
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def normalize_pixels(pixels):
    """Normalize byte RGB to [-1,1] in FP32 before the model's BF16 boundary."""
    return pixels.float().div(255).sub(0.5).div(0.5)


def text_cfg(positive, negative, scale):
    """Negative + scale * (positive - negative); 1 selects positive, 0 negative."""
    if scale == 1.0:
        return positive.float()
    if scale == 0.0:
        return negative.float()
    return negative.float() + scale * (positive.float() - negative.float())


def model_autocast(model):
    """Use BF16 on CUDA and for CPU BF16 models; keep FP32 CPU checks native."""
    parameter = next(model.parameters())
    return torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                          enabled=parameter.is_cuda or parameter.dtype == torch.bfloat16)


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    # interpolate translation
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # interpolate rotation
    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    # ensure there is no sudden change in qw
    quats = src_quat_vec.as_quat().copy()  # [N, 4]
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


# Copyright 2024-2025 The Alibaba Wan Team Authors. Native TI2V sizing.
def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def block_lengths(frames, block_size):
    """Return a complete AR partition for one latent timeline."""
    if type(frames) is not int or frames <= 0:
        raise ValueError("Latent frame count must be a positive integer")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("AR block size must be a positive integer")
    if frames % block_size == 0:
        return (block_size,) * (frames // block_size)
    raise ValueError(
        f"{frames} latent frames must contain complete {block_size}-frame AR blocks"
    )


def block_starts(lengths):
    """Return each block's start and the final exclusive boundary."""
    starts = []
    cursor = 0
    for length in lengths:
        if type(length) is not int or length <= 0:
            raise ValueError("AR block lengths must be positive integers")
        starts.append(cursor)
        cursor += length
    return tuple(starts), cursor


def read_video_frames(video_path, start_frame=0, frame_count=None, *, frame_indices=None, transform=None):
    """Read consecutive uint8 RGB [F,C,H,W]; return None for an incomplete span."""
    import av

    if start_frame < 0 or (frame_count is not None and frame_count <= 0):
        raise ValueError("start_frame and frame_count are invalid")
    if frame_indices is None and frame_count is not None:
        frame_indices = np.arange(start_frame, start_frame + frame_count)
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices)
        if (frame_indices.ndim != 1 or not len(frame_indices)
                or frame_indices.dtype.kind not in 'iu' or frame_indices[0] < 0
                or (frame_indices[1:] <= frame_indices[:-1]).any()):
            raise ValueError("Frame indices must be nonnegative, strictly increasing integers")
    frames, source_size = [], None
    with av.open(str(video_path)) as video:
        for index, frame in enumerate(video.decode(video=0)):
            if frame_indices is not None:
                if index != frame_indices[len(frames)]:
                    continue
            elif index < start_frame:
                continue
            pixels = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)
            if source_size is None:
                source_size = tuple(pixels.shape[-2:])
            if tuple(pixels.shape[-2:]) != source_size:
                raise ValueError(f"Video resolution changed: {video_path}")
            frames.append(transform(pixels) if transform is not None else pixels)
            if frame_indices is not None and len(frames) == len(frame_indices):
                break
    if not frames or (frame_indices is not None and len(frames) != len(frame_indices)):
        return None
    return torch.stack(frames)


def image_pixels(path):
    from PIL import Image

    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1)


def image_transform(source_size, size, mode='fit'):
    """Return the exact PIL crop box and source-pixel -> output-pixel matrix."""
    source_h, source_w = source_size
    height, width = size
    if min(source_h, source_w, height, width) <= 0 or mode not in ('fit', 'resize'):
        raise ValueError("Positive image sizes and resize_mode=fit or resize are required")
    crop_w, crop_h = float(source_w), float(source_h)
    if mode == 'fit':
        # Keep one crop dimension exactly at the source boundary. Computing
        # both from a floating-point scale can exceed that boundary by an
        # epsilon, which PIL rejects as a negative box offset.
        if width * source_h >= height * source_w:
            crop_h = min(float(source_h), height * source_w / width)
        else:
            crop_w = min(float(source_w), width * source_h / height)
    left, top = (source_w - crop_w) / 2, (source_h - crop_h) / 2
    sx, sy = width / crop_w, height / crop_h
    matrix = np.array([[sx, 0, -left * sx], [0, sy, -top * sy], [0, 0, 1]])
    return (left, top, left + crop_w, top + crop_h), matrix


def resize_pixels(pixels, size, mode='fit', *, resample='lanczos'):
    from PIL import Image

    filters = {'lanczos': Image.Resampling.LANCZOS, 'bilinear': Image.Resampling.BILINEAR,
               'bicubic': Image.Resampling.BICUBIC}
    if resample not in filters:
        raise ValueError(f'Unsupported image resampling: {resample}')

    box, _ = image_transform(pixels.shape[-2:], size, mode)
    if tuple(pixels.shape[-2:]) == tuple(size):
        return pixels
    image = Image.fromarray(pixels.permute(1, 2, 0).numpy())
    image = image.resize((size[1], size[0]), filters[resample], box=box)
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


def chunk_color_correct(video, strength, block_frames=3, *, condition_frames=1, state=None):
    """Official chunk-relative Lab correction; latent outputs stay unmodified."""
    import cv2
    import numpy as np
    first = 4 * (block_frames - condition_frames) + 1
    pixels = video.permute(0, 1, 3, 4, 2).mul(255).cpu()
    output = pixels.clone()
    state = {} if state is None else state
    if not state:
        reference = pixels[0, 0].numpy().astype(np.float64) / 255
        reference = cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_RGB2LAB)
        mean, std = reference.mean((0, 1)), reference.std((0, 1))
        state.update(frames=0, stats=[(mean.copy(), std.copy()) for _ in pixels],
                     previous=[None] * len(pixels))
    if len(state['stats']) != len(pixels):
        raise ValueError("A color correction stream must keep the same batch size")
    for b in range(len(pixels)):
        mean, std = state['stats'][b]
        for t in range(pixels.shape[1]):
            position = state['frames'] + t
            if position == first or (position > first and (position-first) % (4 * block_frames) == 0):
                anchor = output[b, t-1] if t else state['previous'][b]
                anchor = anchor.numpy().astype(np.float64) / 255
                anchor = cv2.cvtColor(anchor.astype(np.float32), cv2.COLOR_RGB2LAB)
                mean, std = anchor.mean((0, 1)), anchor.std((0, 1))
            rgb = pixels[b, t].numpy().astype(np.float64) / 255
            lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
            current_mean, current_std = lab.mean((0, 1)), lab.std((0, 1))
            for channel in range(3):
                lab[..., channel] = ((lab[..., channel] - current_mean[channel]) *
                                     (std[channel] / current_std[channel]) + mean[channel]
                                     if current_std[channel] > 1e-6 else mean[channel])
            corrected = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
            blended = (1-strength) * rgb + strength * corrected.astype(np.float64)
            output[b, t] = torch.from_numpy((blended * 255).clip(0, 255)).to(output.dtype)
        state['stats'][b] = (mean, std)
        state['previous'][b] = output[b, -1].clone()
    state['frames'] += pixels.shape[1]
    return output.clamp(0, 255).div(255).permute(0, 1, 4, 2, 3).to(video.device)


def parse_generation_key(path):
    match = re.fullmatch(r"(\d+)-(\d+)(?:_latent)?", path.stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))

def find_generated_files(directory):
    """Index native outputs by (prompt index, sample index) without loading them."""
    generated = {}
    mp4_files = {}
    for path in sorted(Path(directory).glob("*.mp4")):
        key = parse_generation_key(path)
        if key is not None:
            mp4_files[key] = path
            generated[key] = ("mp4", path)
    for path in sorted(Path(directory).glob("*_latent.pt")):
        key = parse_generation_key(path)
        # MP4 is the canonical rendered output and preserves an external
        # condition that cannot be reconstructed from target latent alone.
        if key is not None and key not in generated:
            generated[key] = ("latent", path)
    return generated, mp4_files

def init_distributed(device="cuda"):
    """Initialize torchrun ranks while preserving direct single-process use."""
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    device = torch.device(device)
    if "RANK" not in os.environ:
        return 0, 1, device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    init_kwargs = {"timeout": timedelta(minutes=10)}
    if device.type == "cuda":
        os.environ.setdefault("NCCL_CROSS_NIC", "1")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        init_kwargs.update(backend="nccl", device_id=device)
    else:
        init_kwargs["backend"] = "gloo"
    dist.init_process_group(**init_kwargs)
    return dist.get_rank(), dist.get_world_size(), device


def _execution_device(device):
    """Resolve implicit CUDA indices before loading or reusing device resources."""
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None and torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
    return device

@contextmanager
def inference_rng(seed=None, *, device=None):
    """Bind the execution device and isolate RNG; omit device to protect all GPUs.

    Explicit sample generators still own the native initial/transition noise
    stream. Seeding global RNG handles auxiliary code without depending on the
    training state, checkpoint loading, or the rank assigned to a sample.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = None if device is None else _execution_device(device)
    devices = []
    if (device is None or device.type == 'cuda') and torch.cuda.is_available():
        devices = (list(range(torch.cuda.device_count())) if device is None else
                   [device.index])
    device_scope = (torch.cuda.device(devices[0])
                    if device is not None and devices else nullcontext())
    try:
        with device_scope, torch.random.fork_rng(devices=devices):
            if seed is not None:
                random.seed(int(seed))
                np.random.seed(int(seed) % (2 ** 32))
                torch.random.default_generator.manual_seed(int(seed))
                for index in devices:
                    with torch.cuda.device(index):
                        torch.cuda.manual_seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

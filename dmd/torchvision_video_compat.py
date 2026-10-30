"""Video writer with a torchvision-compatible interface.

Use torchvision when available and imageio with FFmpeg otherwise.
"""

from __future__ import annotations

import numpy as np

try:  # torchvision may still provide it on older installs; prefer the real one.
    from torchvision.io import write_video as _tv_write_video
except Exception:  # pragma: no cover
    _tv_write_video = None


def _to_uint8_thwc(video_array):
    """Coerce a [T, H, W, C] tensor/array (float in [0,255] or uint8) to uint8 ndarray."""
    try:
        import torch

        if isinstance(video_array, torch.Tensor):
            arr = video_array.detach().to("cpu")
            if arr.dtype != torch.uint8:
                arr = arr.clamp(0, 255).round().to(torch.uint8)
            return arr.numpy()
    except ImportError:
        pass

    arr = np.asarray(video_array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    return arr


def write_video(
    filename,
    video_array,
    fps,
    video_codec="libx264",
    options=None,
    audio_array=None,
    audio_fps=None,
    audio_codec=None,
):
    """Drop-in replacement for torchvision.io.write_video.

    Writes a [T, H, W, C] uint8 (or float in [0, 255]) tensor/array to an mp4.
    Audio arguments are accepted for signature compatibility but ignored.
    """
    if _tv_write_video is not None:
        return _tv_write_video(
            filename, video_array, fps, video_codec=video_codec, options=options
        )

    import imageio

    frames = _to_uint8_thwc(video_array)

    # Pass an explicit CRF option to FFmpeg when provided.
    output_params = ["-loglevel", "error", "-pix_fmt", "yuv420p"]
    if options and "crf" in options:
        output_params += ["-crf", str(options["crf"])]

    # macro_block_size=1 preserves the exact frame dimensions (no 16-px rounding).
    writer = imageio.get_writer(
        str(filename),
        fps=float(fps),
        codec=video_codec,
        macro_block_size=1,
        output_params=output_params,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

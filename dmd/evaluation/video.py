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


def video_info(path):
    import cv2
    # Header validation needs one decoder thread, including on CPU-heavy trainers.
    capture = (cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
               if hasattr(cv2, 'CAP_PROP_N_THREADS') else cv2.VideoCapture(str(path)))
    try:
        return {'frame_count': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': float(capture.get(cv2.CAP_PROP_FPS)),
                'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        capture.release()


def valid_video(path, frames, fps=16, width=832, height=480):
    from pathlib import Path
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        info = video_info(path)
        return (info['frame_count'] == frames and abs(info['fps'] - fps) < .01
                and info['width'] == width and info['height'] == height)
    except (OSError, ValueError):
        return False


def write_video_atomic(path, frames, fps=16, options=None):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.stem + '.partial.mp4')
    write_video(str(temporary), frames, fps=fps, options=options)
    if not valid_video(temporary, len(frames), fps, int(frames.shape[2]), int(frames.shape[1])):
        raise ValueError(f'Incomplete video: {temporary}')
    # Flush the completed file before making it visible to resume checks.
    import os
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)

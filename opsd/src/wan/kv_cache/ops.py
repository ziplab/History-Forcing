# Adapted from cvlab-kaist/WorldKV, Apache-2.0.
"""Temporal RoPE shifts for camera-attention cache views."""

import torch


def _rope_time_delta_mul_(k_chunk, freqs, delta_frames):
    """In-place time-axis RoPE shift for already-roped keys.

    k_chunk: [B, L, H, D] where D is head_dim (real-valued RoPE-packed).
    freqs  : complex RoPE freqs, shape [max_seq_len, D/2].
    delta_frames: number of frames to shift (positive: forward, negative: backward).

    Only the *time* channel portion is rotated by exp(i * w * delta_frames).
    """
    if delta_frames == 0:
        return

    b, l, h, d = k_chunk.shape
    assert d % 2 == 0
    c = d // 2
    t_c = c - 2 * (c // 3)
    h_c = c // 3
    w_c = c // 3

    freqs_t, _, _ = freqs.split([t_c, h_c, w_c], dim=1)

    shift = abs(int(delta_frames))
    max_pos = freqs_t.shape[0] - 1
    if shift > max_pos:
        shift = max_pos

    mult = freqs_t[shift] if delta_frames >= 0 else torch.conj(freqs_t[shift])
    mult = mult.view(1, 1, 1, t_c)

    time_ri = k_chunk[..., : 2 * t_c]
    time_cx = torch.view_as_complex(time_ri.to(torch.float64).reshape(-1, t_c, 2))
    time_cx = time_cx * mult.to(time_cx.dtype)
    time_ri_new = torch.view_as_real(time_cx).reshape(b, l, h, t_c, 2).flatten(-2)
    time_ri.copy_(time_ri_new.to(time_ri.dtype))

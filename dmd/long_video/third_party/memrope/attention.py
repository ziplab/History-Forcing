"""MemRoPE attention and rotary position embeddings."""
# Source: https://github.com/YoungRaeKimm/MemRoPE
# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from wan.modules.attention import attention
from wan.modules.model import WanRMSNorm, rope_apply
from torch.nn.attention.flex_attention import flex_attention
import torch.nn as nn
import torch
import math

def causal_rope_apply_with_spatial_indices(
    x, grid_sizes, freqs, sink_tokens, num_rolled_tokens, num_recent_tokens, num_new_tokens,
    frame_seqlen, compressed_temporal_indices, compressed_spatial_indices, global_end_frame, use_block_rope,
    local_attn_size, sink_frames, recent_frames
):
    """
    Apply RoPE to a [Sink] + [Compressed] + [Recent] + [New] cache.

    Block-Relativistic RoPE:
    - Sink, compressed and recent regions use consecutive frame positions from 0.
    - New tokens occupy [local_attn_size - new_frames, ..., local_attn_size - 1],
      leaving a gap after the recent region when the window is not full.

    Args:
        x: [B, L, H, D] - cache tokens (sink + compressed + recent + new)
        num_rolled_tokens: number of compressed tokens
        num_recent_tokens: number of recent tokens (always preserved)
        compressed_temporal_indices: [B, num_rolled_tokens] - original frame index for each compressed token
        compressed_spatial_indices: [B, num_rolled_tokens] - original spatial position (0 to frame_seqlen-1) for each compressed token

    Temporal and spatial frequencies are gathered for all tokens together.
    """
    b, total_tokens, n, c2 = x.shape
    c = c2 // 2
    device = x.device
    f, h, w = int(grid_sizes[0, 0].item()), int(grid_sizes[0, 1].item()), int(grid_sizes[0, 2].item())

    # Split freqs
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freq_t, freq_h, freq_w = freqs_split[0], freqs_split[1], freqs_split[2]
    max_t_idx = freq_t.shape[0] - 1
    max_h_idx = freq_h.shape[0] - 1
    max_w_idx = freq_w.shape[0] - 1

    # Calculate frame counts
    new_frame_count = num_new_tokens // frame_seqlen if num_new_tokens > 0 else 0
    recent_frame_count = num_recent_tokens // frame_seqlen if num_recent_tokens > 0 else 0
    compressed_frame_count = (num_rolled_tokens + frame_seqlen - 1) // frame_seqlen if num_rolled_tokens > 0 else 0

    if use_block_rope:
        # Block-Relativistic RoPE: sequential indices starting from 0
        sink_temporal_base = 0
        compressed_temporal_base = sink_frames
        recent_temporal_base = sink_frames + compressed_frame_count
        new_temporal_base = local_attn_size - new_frame_count
    else:
        # Growing RoPE: use original global temporal indices
        sink_temporal_base = 0  # Sink frames always at global frames 0..sink_frames-1
        compressed_temporal_base = sink_frames  # Fallback when no compressed_temporal_indices
        recent_temporal_base = global_end_frame - new_frame_count - recent_frame_count
        new_temporal_base = global_end_frame - new_frame_count

    # Build temporal and spatial indices for all tokens at once
    temporal_indices_list = []
    h_indices_list = []
    w_indices_list = []

    # === SINK TOKENS ===
    if sink_tokens > 0:
        sink_token_indices = torch.arange(sink_tokens, device=device)
        sink_frame_indices = sink_token_indices // frame_seqlen
        sink_temporal = (sink_temporal_base + sink_frame_indices).clamp(0, max_t_idx)
        sink_local = sink_token_indices % frame_seqlen
        sink_h = (sink_local // w).clamp(0, max_h_idx)
        sink_w = (sink_local % w).clamp(0, max_w_idx)
        temporal_indices_list.append(sink_temporal)
        h_indices_list.append(sink_h)
        w_indices_list.append(sink_w)

    # === COMPRESSED TOKENS ===
    if num_rolled_tokens > 0:
        comp_token_indices = torch.arange(num_rolled_tokens, device=device)
        if not use_block_rope and compressed_temporal_indices is not None:
            # Growing RoPE: use actual global frame index per token
            comp_temporal = compressed_temporal_indices[0].long().clamp(0, max_t_idx)
        else:
            # Block-relativistic: sequential based on position in compressed region
            comp_temporal = (compressed_temporal_base + comp_token_indices // frame_seqlen).clamp(0, max_t_idx)

        # Spatial: use stored original positions if available
        if compressed_spatial_indices is not None:
            # Use first batch's spatial indices (assume same across batch)
            comp_local = compressed_spatial_indices[0].long()
        else:
            comp_local = comp_token_indices % frame_seqlen
        comp_h = (comp_local // w).clamp(0, max_h_idx)
        comp_w = (comp_local % w).clamp(0, max_w_idx)
        temporal_indices_list.append(comp_temporal)
        h_indices_list.append(comp_h)
        w_indices_list.append(comp_w)

    # === RECENT TOKENS ===
    if num_recent_tokens > 0:
        recent_token_indices = torch.arange(num_recent_tokens, device=device)
        recent_frame_in_region = recent_token_indices // frame_seqlen
        recent_temporal = (recent_temporal_base + recent_frame_in_region).clamp(0, max_t_idx)
        recent_local = recent_token_indices % frame_seqlen
        recent_h = (recent_local // w).clamp(0, max_h_idx)
        recent_w = (recent_local % w).clamp(0, max_w_idx)
        temporal_indices_list.append(recent_temporal)
        h_indices_list.append(recent_h)
        w_indices_list.append(recent_w)

    # === NEW TOKENS ===
    if num_new_tokens > 0:
        new_token_indices = torch.arange(num_new_tokens, device=device)
        new_frame_in_region = new_token_indices // frame_seqlen
        new_temporal = (new_temporal_base + new_frame_in_region).clamp(0, max_t_idx)
        new_local = new_token_indices % frame_seqlen
        new_h = (new_local // w).clamp(0, max_h_idx)
        new_w = (new_local % w).clamp(0, max_w_idx)
        temporal_indices_list.append(new_temporal)
        h_indices_list.append(new_h)
        w_indices_list.append(new_w)

    if not temporal_indices_list:
        return x

    # Concatenate all indices
    all_temporal = torch.cat(temporal_indices_list, dim=0)  # [L]
    all_h = torch.cat(h_indices_list, dim=0)  # [L]
    all_w = torch.cat(w_indices_list, dim=0)  # [L]
    L = all_temporal.shape[0]

    # Gather frequencies for all tokens at once
    freq_temporal = freq_t[all_temporal.long()]  # [L, c_t]
    freq_height = freq_h[all_h.long()]  # [L, c_h]
    freq_width = freq_w[all_w.long()]  # [L, c_w]

    # Concatenate frequencies: [L, c]
    all_freqs = torch.cat([freq_temporal, freq_height, freq_width], dim=-1)  # [L, c]
    all_freqs = all_freqs.unsqueeze(1)  # [L, 1, c]

    # Apply RoPE to x
    # x: [B, L, H, D] -> complex view
    x_view = x[:, :L].to(torch.float64).reshape(b, L, n, -1, 2)
    x_complex = torch.view_as_complex(x_view)  # [B, L, H, c]

    # Broadcast freqs across batch and heads
    result = x_complex * all_freqs.unsqueeze(0)  # [B, L, H, c]
    result = torch.view_as_real(result).flatten(3)  # [B, L, H, D]

    return result.type_as(x)

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, relative_frame_indices=None):
    """
    Apply causal RoPE (Rotary Position Embedding) to input tensor.
    
    Args:
        x: Input tensor of shape [B, L, num_heads, head_dim]
        grid_sizes: Tensor of shape [B, 3] containing (F, H, W)
        freqs: RoPE frequencies
        start_frame: Starting frame index for sequential RoPE (default: 0)
        relative_frame_indices: Optional tensor of shape [F] specifying explicit frame indices
                               for Block-Relativistic RoPE. If provided, overrides start_frame.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # Use relative_frame_indices if provided (Block-Relativistic RoPE),
        # otherwise use sequential indices starting from start_frame
        if relative_frame_indices is not None:
            # relative_frame_indices should be a tensor of shape [f] with explicit frame indices
            frame_indices = relative_frame_indices.long()
            freqs_temporal = freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1)
        else:
            freqs_temporal = freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1)
        
        freqs_i = torch.cat([
            freqs_temporal,
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 recent_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 use_block_rope=True,
                 compression_method='eviction',
                 ema_alpha_long=0.01,
                 ema_alpha_short=0.1,
                 ema_adaptive=False):
        """
        Args:
            sink_size: number of sink frames to preserve at the beginning
            recent_size: number of recent frames to preserve
            compression_method: 'eviction' (default) or 'ema'
                - eviction: simple FIFO eviction of oldest tokens
                - ema: [Sink] + [Long-term EMA] + [Short-term EMA] + [Recent] + [New]
                  Two EMA memories with different update rates
            ema_alpha_long: EMA update rate for long-term memory (small = slow update, default 0.01)
            ema_alpha_short: EMA update rate for short-term memory (large = fast update, default 0.1)
            ema_adaptive: if True, use per-token motion-based adaptive alpha (default False)
        """
        assert dim % num_heads == 0
        assert compression_method in ['eviction', 'ema']
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_block_rope = use_block_rope
        self.compression_method = compression_method
        self.ema_alpha_long = ema_alpha_long
        self.ema_alpha_short = ema_alpha_short
        self.ema_adaptive = ema_adaptive
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # Teacher forcing concatenates clean and noisy tokens.
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # Clean and noisy tokens share temporal RoPE positions.
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            num_new_frames = grid_sizes[0][0].item()  # F from grid_sizes
            
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            sink_frames = self.sink_size
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            kv_cache_frames = kv_cache_size // frame_seqlen
            
            # Determine if we're in Block-Relativistic RoPE mode (cache is full or rolling)
            is_rolling_mode = self.local_attn_size != -1 and (
                kv_cache["local_end_index"].item() + num_new_tokens > kv_cache_size
            )
            
            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # === ROLLING MODE ===
                # Calculate the number of tokens to evict/compress
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_evicted_frames = num_evicted_tokens // frame_seqlen

                # Create temporary k, v for computation - store UN-ROPED K
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()

                if self.compression_method == 'ema':
                    # === EMA MEMORY COMPRESSION ===
                    # Cache structure: [Sink 3] + [Long-term EMA 1] + [Short-term EMA 1] + [Recent 4] + [New 3] = 12
                    # Long-term EMA: slow update (small alpha) - retains distant past
                    # Short-term EMA: fast update (large alpha) - captures recent trends
                    #
                    # If ema_adaptive=True: per-token alpha based on motion
                    #   motion[pos] = ||evicted[pos] - old_ema[pos]||
                    #   high motion → large alpha, low motion → small alpha

                    num_ema_frames = 2  # 1 long-term + 1 short-term
                    ema_tokens = num_ema_frames * frame_seqlen
                    current_local_end = kv_cache["local_end_index"].item()

                    # Recent fills the rest: total - sink - ema - new
                    dynamic_recent_frames = self.local_attn_size - self.sink_size - num_ema_frames - num_new_frames
                    recent_tokens = dynamic_recent_frames * frame_seqlen

                    # EMA positions (after sink)
                    ema_long_start = sink_tokens
                    ema_long_end = sink_tokens + frame_seqlen
                    ema_short_start = ema_long_end
                    ema_short_end = ema_short_start + frame_seqlen

                    # Recent starts after EMA
                    recent_start = ema_short_end
                    recent_end = current_local_end
                    actual_recent_tokens = recent_end - recent_start

                    # Always preserve sink frames
                    temp_k[:, :sink_tokens] = kv_cache["k"][:, :sink_tokens].clone()
                    temp_v[:, :sink_tokens] = kv_cache["v"][:, :sink_tokens].clone()

                    # Variables to pass to _apply_cache_updates
                    alpha_long_for_cache = self.ema_alpha_long  # scalar or tensor
                    alpha_short_for_cache = self.ema_alpha_short
                    evicted_k_for_cache = None
                    evicted_v_for_cache = None

                    if is_recompute:
                        # Reuse the committed EMA and recent regions during recomputation.
                        temp_k[:, ema_long_start:ema_short_end] = kv_cache["k"][:, ema_long_start:ema_short_end].clone()
                        temp_v[:, ema_long_start:ema_short_end] = kv_cache["v"][:, ema_long_start:ema_short_end].clone()

                        local_start_index = current_local_end - num_new_tokens
                        local_end_index = current_local_end
                        num_kept_recent = local_start_index - ema_short_end

                        if num_kept_recent > 0:
                            temp_k[:, ema_short_end:local_start_index] = kv_cache["k"][:, ema_short_end:local_start_index].clone()
                            temp_v[:, ema_short_end:local_start_index] = kv_cache["v"][:, ema_short_end:local_start_index].clone()

                        temp_k[:, local_start_index:local_end_index] = k
                        temp_v[:, local_start_index:local_end_index] = v
                    else:
                        # Build the next cache layout and update EMA memory.
                        ema_initialized = "ema_initialized" in kv_cache and kv_cache["ema_initialized"]

                        if not ema_initialized:
                            # First time: initialize EMA
                            if num_evicted_tokens > 0:
                                evicted_k = kv_cache["k"][:, recent_start:recent_start + num_evicted_tokens]
                                evicted_v = kv_cache["v"][:, recent_start:recent_start + num_evicted_tokens]
                                if self.ema_adaptive and num_evicted_tokens >= frame_seqlen:
                                    # Per-position mean: preserves spatial structure
                                    num_evicted_frames = num_evicted_tokens // frame_seqlen
                                    evicted_k_mean = evicted_k[:, :num_evicted_frames * frame_seqlen].view(
                                        b, num_evicted_frames, frame_seqlen, n, d).mean(dim=1)
                                    evicted_v_mean = evicted_v[:, :num_evicted_frames * frame_seqlen].view(
                                        b, num_evicted_frames, frame_seqlen, n, d).mean(dim=1)
                                else:
                                    # Average over all evicted tokens.
                                    evicted_k_mean = evicted_k.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                    evicted_v_mean = evicted_v.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                temp_k[:, ema_long_start:ema_long_end] = evicted_k_mean
                                temp_v[:, ema_long_start:ema_long_end] = evicted_v_mean
                                temp_k[:, ema_short_start:ema_short_end] = evicted_k_mean
                                temp_v[:, ema_short_start:ema_short_end] = evicted_v_mean
                                evicted_k_for_cache = evicted_k_mean
                                evicted_v_for_cache = evicted_v_mean
                            else:
                                temp_k[:, ema_long_start:ema_long_end] = kv_cache["k"][:, recent_start:recent_start + frame_seqlen].clone()
                                temp_v[:, ema_long_start:ema_long_end] = kv_cache["v"][:, recent_start:recent_start + frame_seqlen].clone()
                                temp_k[:, ema_short_start:ema_short_end] = kv_cache["k"][:, recent_start:recent_start + frame_seqlen].clone()
                                temp_v[:, ema_short_start:ema_short_end] = kv_cache["v"][:, recent_start:recent_start + frame_seqlen].clone()
                        else:
                            # Update EMA with evicted frames
                            old_ema_long_k = kv_cache["k"][:, ema_long_start:ema_long_end].clone()
                            old_ema_long_v = kv_cache["v"][:, ema_long_start:ema_long_end].clone()
                            old_ema_short_k = kv_cache["k"][:, ema_short_start:ema_short_end].clone()
                            old_ema_short_v = kv_cache["v"][:, ema_short_start:ema_short_end].clone()

                            if num_evicted_tokens > 0:
                                evicted_k = kv_cache["k"][:, recent_start:recent_start + num_evicted_tokens]
                                evicted_v = kv_cache["v"][:, recent_start:recent_start + num_evicted_tokens]

                                if self.ema_adaptive and num_evicted_tokens >= frame_seqlen:
                                    # Per-position mean: preserves spatial structure
                                    num_evicted_frames = num_evicted_tokens // frame_seqlen
                                    evicted_k_mean = evicted_k[:, :num_evicted_frames * frame_seqlen].view(
                                        b, num_evicted_frames, frame_seqlen, n, d).mean(dim=1)
                                    evicted_v_mean = evicted_v[:, :num_evicted_frames * frame_seqlen].view(
                                        b, num_evicted_frames, frame_seqlen, n, d).mean(dim=1)

                                    # Long-term EMA: uniform alpha (stable global scene summary)
                                    alpha_long = self.ema_alpha_long

                                    # Short-term EMA: adaptive per-token alpha based on motion
                                    motion_short = (evicted_k_mean - old_ema_short_k).norm(dim=-1).mean(dim=-1)  # [B, frame_seqlen]
                                    motion_short_norm = motion_short / (motion_short.max(dim=-1, keepdim=True).values + 1e-8)  # [0, 1]

                                    alpha_short_min = self.ema_alpha_short * 0.1
                                    alpha_short_max = self.ema_alpha_short * 5.0
                                    alpha_short = (alpha_short_min + motion_short_norm * (alpha_short_max - alpha_short_min)).unsqueeze(-1).unsqueeze(-1)  # [B, frame_seqlen, 1, 1]
                                else:
                                    # Global mean with scalar EMA update rates.
                                    evicted_k_mean = evicted_k.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                    evicted_v_mean = evicted_v.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                    alpha_long = self.ema_alpha_long
                                    alpha_short = self.ema_alpha_short

                                # EMA update (works with both scalar and per-token alpha)
                                new_ema_long_k = alpha_long * evicted_k_mean + (1 - alpha_long) * old_ema_long_k
                                new_ema_long_v = alpha_long * evicted_v_mean + (1 - alpha_long) * old_ema_long_v
                                new_ema_short_k = alpha_short * evicted_k_mean + (1 - alpha_short) * old_ema_short_k
                                new_ema_short_v = alpha_short * evicted_v_mean + (1 - alpha_short) * old_ema_short_v

                                temp_k[:, ema_long_start:ema_long_end] = new_ema_long_k
                                temp_v[:, ema_long_start:ema_long_end] = new_ema_long_v
                                temp_k[:, ema_short_start:ema_short_end] = new_ema_short_k
                                temp_v[:, ema_short_start:ema_short_end] = new_ema_short_v

                                # Store for _apply_cache_updates
                                alpha_long_for_cache = alpha_long
                                alpha_short_for_cache = alpha_short
                                evicted_k_for_cache = evicted_k_mean
                                evicted_v_for_cache = evicted_v_mean
                            else:
                                # No evicted tokens, keep old EMA
                                temp_k[:, ema_long_start:ema_long_end] = old_ema_long_k
                                temp_v[:, ema_long_start:ema_long_end] = old_ema_long_v
                                temp_k[:, ema_short_start:ema_short_end] = old_ema_short_k
                                temp_v[:, ema_short_start:ema_short_end] = old_ema_short_v

                        # FIFO shift Recent
                        k_recent = kv_cache["k"][:, recent_start + num_evicted_tokens:recent_end].clone()
                        v_recent = kv_cache["v"][:, recent_start + num_evicted_tokens:recent_end].clone()

                        num_kept_recent = k_recent.shape[1]
                        write_pos = ema_short_end
                        temp_k[:, write_pos:write_pos + num_kept_recent] = k_recent
                        temp_v[:, write_pos:write_pos + num_kept_recent] = v_recent
                        write_pos += num_kept_recent

                        local_end_index = write_pos + num_new_tokens
                        local_start_index = write_pos

                        temp_k[:, local_start_index:local_end_index] = k
                        temp_v[:, local_start_index:local_end_index] = v

                    # Apply RoPE (Block-Relativistic)
                    num_cache_frames = local_end_index // frame_seqlen
                    cache_grid_sizes = grid_sizes.clone()
                    cache_grid_sizes[0, 0] = num_cache_frames

                    query_relative_indices = torch.arange(
                        self.local_attn_size - num_new_frames,
                        self.local_attn_size,
                        device=q.device
                    )
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
                    ).type_as(v)

                    cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
                    roped_temp_k = causal_rope_apply(
                        temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                        cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices
                    ).type_as(v)

                    # Compute temporal/spatial indices for new tokens
                    new_token_positions = torch.arange(num_new_tokens, device=q.device)
                    new_temporal_indices = (current_start + new_token_positions) // frame_seqlen
                    new_spatial_indices = (current_start + new_token_positions) % frame_seqlen

                    cache_update_info = {
                        "action": "ema",
                        "sink_tokens": sink_tokens,
                        "ema_tokens": ema_tokens,
                        "local_start_index": local_start_index,
                        "local_end_index": local_end_index,
                        "write_start_index": local_start_index,
                        "write_end_index": local_end_index,
                        "new_k": k,
                        "new_v": v,
                        "new_q": q,
                        "new_temporal_indices": new_temporal_indices,
                        "new_spatial_indices": new_spatial_indices,
                        "current_end": current_end,
                        "is_recompute": is_recompute,
                        "num_evicted_tokens": num_evicted_tokens,
                        "alpha_long_tensor": alpha_long_for_cache,
                        "alpha_short_tensor": alpha_short_for_cache,
                        "evicted_k_mean": evicted_k_for_cache,
                        "evicted_v_mean": evicted_v_for_cache,
                        "ema_adaptive": self.ema_adaptive,
                    }

                else:
                    # === SIMPLE EVICTION (FIFO) ===
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                    # Compute updated local indices
                    local_end_index = kv_cache["local_end_index"].item() + current_end - \
                        kv_cache["global_end_index"].item() - num_evicted_tokens
                    local_start_index = local_end_index - num_new_tokens

                    # Apply rolling update to the temporary cache
                    temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                    # Insert new key/value into the temporary cache (UN-ROPED K!)
                    write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                    roped_offset = max(0, write_start_index - local_start_index)
                    write_len = max(0, local_end_index - write_start_index)
                    if write_len > 0:
                        temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                        temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                    # === Block-Relativistic RoPE for FIFO eviction ===
                    num_cache_frames = local_end_index // frame_seqlen
                    cache_grid_sizes = grid_sizes.clone()
                    cache_grid_sizes[0, 0] = num_cache_frames

                    # Query: at end of window [local_attn_size - num_new_frames, ..., local_attn_size - 1]
                    query_relative_indices = torch.arange(
                        self.local_attn_size - num_new_frames,
                        self.local_attn_size,
                        device=q.device
                    )
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
                    ).type_as(v)

                    # Cache: [0, 1, 2, ..., num_cache_frames - 1]
                    cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
                    roped_temp_k = causal_rope_apply(
                        temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                        cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices
                    ).type_as(v)

                    # Compute temporal/spatial indices for new tokens
                    new_token_global_start = current_start + roped_offset
                    new_token_positions = torch.arange(write_len, device=q.device)
                    new_temporal_indices = (new_token_global_start + new_token_positions) // frame_seqlen
                    new_spatial_indices = (new_token_global_start + new_token_positions) % frame_seqlen

                    # Cache update info for eviction - store UN-ROPED K!
                    cache_update_info = {
                        "action": "roll_and_insert",
                        "sink_tokens": sink_tokens,
                        "num_rolled_tokens": num_rolled_tokens,
                        "num_evicted_tokens": num_evicted_tokens,
                        "local_start_index": local_start_index,
                        "local_end_index": local_end_index,
                        "write_start_index": write_start_index,
                        "write_end_index": local_end_index,
                        "new_k": k[:, roped_offset:roped_offset + write_len],
                        "new_v": v[:, roped_offset:roped_offset + write_len],
                        "new_q": q[:, roped_offset:roped_offset + write_len],
                        "new_temporal_indices": new_temporal_indices,  # [write_len]
                        "new_spatial_indices": new_spatial_indices,    # [write_len]
                        "current_end": current_end,
                        "is_recompute": is_recompute
                    }
            else:
                # === DIRECT INSERT MODE ===
                # Before cache is full, we can still use relative indices that grow sequentially
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation
                temp_k = kv_cache["k"].clone()  # UN-ROPED K
                temp_v = kv_cache["v"].clone()
                
                # Protect sink_tokens only during recomputation
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    # Store UN-ROPED K in cache
                    temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # === RoPE with relative temporal indices ===
                # Current frame position in the window
                current_frame_in_window = local_start_index // frame_seqlen

                # Query: apply RoPE with relative frame indices
                query_relative_indices = torch.arange(
                    current_frame_in_window,
                    current_frame_in_window + num_new_frames,
                    device=q.device
                )
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
                ).type_as(v)

                # Cached K: apply RoPE dynamically
                num_cache_frames = local_end_index // frame_seqlen
                cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)

                cache_grid_sizes = grid_sizes.clone()
                cache_grid_sizes[0, 0] = num_cache_frames

                roped_temp_k = causal_rope_apply(
                    temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                    cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices
                ).type_as(v)

                # Compute temporal/spatial indices for new tokens
                new_token_global_start = current_start + roped_offset
                new_token_positions = torch.arange(write_len, device=q.device)
                new_temporal_indices = (new_token_global_start + new_token_positions) // frame_seqlen
                new_spatial_indices = (new_token_global_start + new_token_positions) % frame_seqlen

                # Save cache update info - store UN-ROPED K!
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": k[:, roped_offset:roped_offset + write_len],  # UN-ROPED K!
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "new_q": q[:, roped_offset:roped_offset + write_len],
                    "new_temporal_indices": new_temporal_indices,  # [write_len]
                    "new_spatial_indices": new_spatial_indices,    # [write_len]
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }

            # Use roped K for attention computation
            # Limit V to same range as roped K (which has local_end_index tokens)
            temp_v_active = temp_v[:, :local_end_index]

            if sink_tokens > 0 and local_end_index > sink_tokens:
                # Concatenate sink tokens and local window tokens
                local_budget = self.max_attention_size - sink_tokens
                k_sink = roped_temp_k[:, :sink_tokens]
                v_sink = temp_v_active[:, :sink_tokens]
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local = roped_temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v_active[:, local_start_for_window:local_end_index]
                    k_cat = torch.cat([k_sink, k_local], dim=1)
                    v_cat = torch.cat([v_sink, v_local], dim=1)
                else:
                    k_cat = k_sink
                    v_cat = v_sink

                x = attention(
                    roped_query,
                    k_cat,
                    v_cat,
                    deterministic=True
                )
            else:
                window_start = max(0, local_end_index - self.max_attention_size)

                x = attention(
                    roped_query,
                    roped_temp_k[:, window_start:local_end_index],
                    temp_v_active[:, window_start:local_end_index],
                    deterministic=True
                )

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x

def _apply_cache_updates(self, kv_cache, cache_update_infos):
    """
        Applies cache updates collected from multiple blocks.
        
        For Block-Relativistic RoPE, this stores UN-ROPED K values in the cache.
        RoPE is applied dynamically during attention based on the token's current
        relative position in the sliding window.
        
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
    for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
        if update_info is not None:
            cache = kv_cache[block_index]
            if update_info['action'] == 'roll_and_insert':
                sink_tokens = update_info['sink_tokens']
                num_rolled_tokens = update_info['num_rolled_tokens']
                num_evicted_tokens = update_info['num_evicted_tokens']
                local_start_index = update_info['local_start_index']
                local_end_index = update_info['local_end_index']
                write_start_index = update_info.get('write_start_index', local_start_index)
                write_end_index = update_info.get('write_end_index', local_end_index)
                new_k = update_info['new_k']
                new_v = update_info['new_v']
                new_q = update_info.get('new_q')
                new_temporal_indices = update_info.get('new_temporal_indices')
                new_spatial_indices = update_info.get('new_spatial_indices')
                cache['k'][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache['k'][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                cache['v'][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache['v'][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                if 'q' in cache:
                    cache['q'][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache['q'][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                if 'token_temporal_indices' in cache:
                    cache['token_temporal_indices'][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache['token_temporal_indices'][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                if 'token_spatial_indices' in cache:
                    cache['token_spatial_indices'][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache['token_spatial_indices'][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                if write_end_index > write_start_index and new_k.shape[1] == write_end_index - write_start_index:
                    cache['k'][:, write_start_index:write_end_index] = new_k
                    cache['v'][:, write_start_index:write_end_index] = new_v
                    if new_q is not None and 'q' in cache:
                        cache['q'][:, write_start_index:write_end_index] = new_q
                    if new_temporal_indices is not None and 'token_temporal_indices' in cache:
                        cache['token_temporal_indices'][:, write_start_index:write_end_index] = new_temporal_indices
                    if new_spatial_indices is not None and 'token_spatial_indices' in cache:
                        cache['token_spatial_indices'][:, write_start_index:write_end_index] = new_spatial_indices
            elif update_info['action'] == 'ema':
                sink_tokens = update_info['sink_tokens']
                ema_tokens = update_info['ema_tokens']
                write_start_index = update_info['write_start_index']
                write_end_index = update_info['write_end_index']
                local_start_index = update_info['local_start_index']
                new_k = update_info['new_k']
                new_v = update_info['new_v']
                new_q = update_info.get('new_q')
                new_temporal_indices = update_info.get('new_temporal_indices')
                new_spatial_indices = update_info.get('new_spatial_indices')
                is_recompute = update_info.get('is_recompute', False)
                num_evicted_tokens = update_info.get('num_evicted_tokens', 0)
                alpha_long = update_info.get('alpha_long_tensor', 0.01)
                alpha_short = update_info.get('alpha_short_tensor', 0.1)
                evicted_k_mean_from_fwd = update_info.get('evicted_k_mean')
                evicted_v_mean_from_fwd = update_info.get('evicted_v_mean')
                is_adaptive = update_info.get('ema_adaptive', False)
                frame_seqlen = ema_tokens // 2
                ema_long_start = sink_tokens
                ema_long_end = sink_tokens + frame_seqlen
                ema_short_start = ema_long_end
                ema_short_end = ema_short_start + frame_seqlen
                recent_start = ema_short_end
                if not is_recompute:
                    current_local_end = cache['local_end_index'].item()
                    ema_initialized = 'ema_initialized' in cache and cache['ema_initialized']
                    if not ema_initialized:
                        if evicted_k_mean_from_fwd is not None:
                            cache['k'][:, ema_long_start:ema_long_end] = evicted_k_mean_from_fwd
                            cache['v'][:, ema_long_start:ema_long_end] = evicted_v_mean_from_fwd
                            cache['k'][:, ema_short_start:ema_short_end] = evicted_k_mean_from_fwd
                            cache['v'][:, ema_short_start:ema_short_end] = evicted_v_mean_from_fwd
                        elif num_evicted_tokens > 0:
                            evicted_k = cache['k'][:, recent_start:recent_start + num_evicted_tokens]
                            evicted_v = cache['v'][:, recent_start:recent_start + num_evicted_tokens]
                            evicted_k_mean = evicted_k.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                            evicted_v_mean = evicted_v.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                            cache['k'][:, ema_long_start:ema_long_end] = evicted_k_mean
                            cache['v'][:, ema_long_start:ema_long_end] = evicted_v_mean
                            cache['k'][:, ema_short_start:ema_short_end] = evicted_k_mean
                            cache['v'][:, ema_short_start:ema_short_end] = evicted_v_mean
                        else:
                            cache['k'][:, ema_long_start:ema_long_end] = cache['k'][:, recent_start:recent_start + frame_seqlen].clone()
                            cache['v'][:, ema_long_start:ema_long_end] = cache['v'][:, recent_start:recent_start + frame_seqlen].clone()
                            cache['k'][:, ema_short_start:ema_short_end] = cache['k'][:, recent_start:recent_start + frame_seqlen].clone()
                            cache['v'][:, ema_short_start:ema_short_end] = cache['v'][:, recent_start:recent_start + frame_seqlen].clone()
                        cache['ema_initialized'] = True
                    elif num_evicted_tokens > 0:
                        old_ema_long_k = cache['k'][:, ema_long_start:ema_long_end].clone()
                        old_ema_long_v = cache['v'][:, ema_long_start:ema_long_end].clone()
                        old_ema_short_k = cache['k'][:, ema_short_start:ema_short_end].clone()
                        old_ema_short_v = cache['v'][:, ema_short_start:ema_short_end].clone()
                        if evicted_k_mean_from_fwd is not None:
                            evicted_k_mean = evicted_k_mean_from_fwd
                            evicted_v_mean = evicted_v_mean_from_fwd
                        else:
                            evicted_k = cache['k'][:, recent_start:recent_start + num_evicted_tokens]
                            evicted_v = cache['v'][:, recent_start:recent_start + num_evicted_tokens]
                            evicted_k_mean = evicted_k.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                            evicted_v_mean = evicted_v.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                        cache['k'][:, ema_long_start:ema_long_end] = alpha_long * evicted_k_mean + (1 - alpha_long) * old_ema_long_k
                        cache['v'][:, ema_long_start:ema_long_end] = alpha_long * evicted_v_mean + (1 - alpha_long) * old_ema_long_v
                        cache['k'][:, ema_short_start:ema_short_end] = alpha_short * evicted_k_mean + (1 - alpha_short) * old_ema_short_k
                        cache['v'][:, ema_short_start:ema_short_end] = alpha_short * evicted_v_mean + (1 - alpha_short) * old_ema_short_v
                    remaining_recent = local_start_index - ema_short_end
                    if remaining_recent > 0 and num_evicted_tokens > 0:
                        cache['k'][:, recent_start:recent_start + remaining_recent] = cache['k'][:, recent_start + num_evicted_tokens:current_local_end].clone()
                        cache['v'][:, recent_start:recent_start + remaining_recent] = cache['v'][:, recent_start + num_evicted_tokens:current_local_end].clone()
                        if 'q' in cache:
                            cache['q'][:, recent_start:recent_start + remaining_recent] = cache['q'][:, recent_start + num_evicted_tokens:current_local_end].clone()
                if write_end_index > write_start_index:
                    cache['k'][:, write_start_index:write_end_index] = new_k
                    cache['v'][:, write_start_index:write_end_index] = new_v
                    if new_q is not None and 'q' in cache:
                        cache['q'][:, write_start_index:write_end_index] = new_q
                    if new_temporal_indices is not None and 'token_temporal_indices' in cache:
                        cache['token_temporal_indices'][:, write_start_index:write_end_index] = new_temporal_indices
                    if new_spatial_indices is not None and 'token_spatial_indices' in cache:
                        cache['token_spatial_indices'][:, write_start_index:write_end_index] = new_spatial_indices
            elif update_info['action'] == 'direct_insert':
                local_start_index = update_info['local_start_index']
                local_end_index = update_info['local_end_index']
                write_start_index = update_info.get('write_start_index', local_start_index)
                write_end_index = update_info.get('write_end_index', local_end_index)
                new_k = update_info['new_k']
                new_v = update_info['new_v']
                new_q = update_info.get('new_q')
                new_temporal_indices = update_info.get('new_temporal_indices')
                new_spatial_indices = update_info.get('new_spatial_indices')
                if write_end_index > write_start_index and new_k.shape[1] == write_end_index - write_start_index:
                    cache['k'][:, write_start_index:write_end_index] = new_k
                    cache['v'][:, write_start_index:write_end_index] = new_v
                    if new_q is not None and 'q' in cache:
                        cache['q'][:, write_start_index:write_end_index] = new_q
                    if new_temporal_indices is not None and 'token_temporal_indices' in cache:
                        cache['token_temporal_indices'][:, write_start_index:write_end_index] = new_temporal_indices
                    if new_spatial_indices is not None and 'token_spatial_indices' in cache:
                        cache['token_spatial_indices'][:, write_start_index:write_end_index] = new_spatial_indices
        is_recompute = False if update_info is None else update_info.get('is_recompute', False)
        if not is_recompute:
            kv_cache[block_index]['global_end_index'].fill_(current_end)
            kv_cache[block_index]['local_end_index'].fill_(local_end_index)

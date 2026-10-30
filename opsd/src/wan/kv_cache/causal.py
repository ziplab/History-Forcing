"""Bounded Wan KV allocation, clean commits, and differentiable history reads."""

from copy import copy
from collections import UserList

import torch


class CachePosition:
    """Host cursors in tokens, excluding any separate reference-image slots.

    An opaque mutable object survives FSDP's reconstruction of input dicts and
    avoids CUDA scalar synchronization in every attention layer. ``rope_base``
    is the token offset used by backends that periodically rebase RoPE.
    """

    def __init__(self):
        self.end = 0
        self.length = 0
        self.rope_base = 0


class CausalCache:
    """One rollout's named self-attention streams and static text context.

    Stream names are the model's forward keyword arguments. The sampler and
    recipe pass them through without knowing which modalities created them.
    Each stream owns its layer tensors; read views have independent storage.
    """

    def __init__(self, streams, cross, *, reference_tokens=0):
        # Cache state is allocated on its model's device. Keep each stream
        # opaque to DDP/FSDP input traversal: rebuilding nested dicts drops
        # committed token positions and text-cache updates between forwards.
        self.streams = {name: UserList(layers) for name, layers in streams.items()}
        self.cross = UserList(cross)
        self.reference_tokens = reference_tokens

    def forward_kwargs(self, *, cache_text=True):
        return {**self.streams, "crossattn_cache": self.cross if cache_text else None}

    def read_only_snapshot(self):
        """Own detached history until backward, independently of later commits.

        Checkpoint callbacks retain these tensors and host cursors. A shallow
        stream copy still aliases the persistent KV buffers.
        """
        def snapshot(value):
            if torch.is_tensor(value):
                return value.detach().clone()
            if isinstance(value, CachePosition):
                return copy(value)
            if isinstance(value, dict):
                return {name: snapshot(item) for name, item in value.items()}
            if isinstance(value, list):
                return [snapshot(item) for item in value]
            if isinstance(value, tuple):
                return tuple(snapshot(item) for item in value)
            return value

        streams = {
            name: [{field: snapshot(value) for field, value in layer.items()}
                   for layer in layers]
            for name, layers in self.streams.items()
        }
        return type(self)(streams, [dict(is_init=False) for _ in self.cross],
                          reference_tokens=self.reference_tokens)

    def seed_from(self, source, frame, tokens_per_frame, *, sink_frames=0,
                  history_frames=0, source_sink_frames=None, rebase_limit=256):
        """Copy a fixed-history read view before the replacement span at ``frame``.

        Only sinks preceding ``frame`` are copied; overlapping sink frames are
        encoded by the subsequent recent commits, avoiding duplicate history.
        ``history_frames`` budgets earlier non-sink KV; None keeps the native
        history. Native sinks excluded by this
        view are not counted again as fixed history. The next commit lands at
        the same absolute time and RoPE base as ``source``. Indexed relative
        RoPE follows the retained physical slots. Reference slots are copied
        because backends re-rope them from their raw bank.
        ``rebase_limit`` rejects frames past a backend's RoPE rebase, which would
        move the stored frame to a different relative position than the source;
        backends without rebasing pass ``None``.
        """
        if type(frame) is not int or frame < 0:
            raise ValueError("seed_from requires a nonnegative start frame")
        if tokens_per_frame <= 0:
            raise ValueError("seed_from needs a positive tokens_per_frame")
        if type(sink_frames) is not int or sink_frames < 0:
            raise ValueError("seed_from requires a nonnegative sink frame count")
        if history_frames is not None and (type(history_frames) is not int or history_frames < 0):
            raise ValueError('seed_from requires a nonnegative history frame count or None')
        if source_sink_frames is None:
            source_sink_frames = sink_frames
        if type(source_sink_frames) is not int or not 0 <= sink_frames <= source_sink_frames:
            raise ValueError('seed_from sinks must be present in the source native sink')
        sink_tokens = min(sink_frames, frame) * tokens_per_frame
        if self is source:
            raise ValueError("seed_from needs a cache with its own buffers")
        if self.reference_tokens != source.reference_tokens:
            raise ValueError("seed_from requires matching reference slot counts")
        if self.streams.keys() != source.streams.keys():
            raise ValueError("seed_from requires matching cache streams")
        if any(layer is origin for name, layers in self.streams.items()
               for layer, origin in zip(layers, source.streams[name])):
            raise ValueError("seed_from needs a cache with its own buffers")
        with torch.no_grad():
            for name, layers in self.streams.items():
                origins = source.streams[name]
                if len(layers) != len(origins):
                    raise ValueError("seed_from requires matching cache layers")
                for layer, origin in zip(layers, origins, strict=True):
                    position, origin_position = layer['position'], origin['position']
                    if ('k_raw' in layer) != ('k_raw' in origin):
                        raise ValueError("seed_from requires matching cache keys")
                    if layer['k'].shape[1] < self.reference_tokens + sink_tokens + tokens_per_frame:
                        raise ValueError("seed_from target has no room for sinks and one recent frame")
                    # A rebase would place the rewritten frame at a different
                    # relative position than the source cache reads it from.
                    base = origin_position.rope_base // tokens_per_frame
                    if rebase_limit is not None and (
                        frame - base + 1 > rebase_limit or frame - base < 0
                    ):
                        raise ValueError(
                            "seed_from cannot rewind across a RoPE rebase at frame "
                            f"{frame} (base {base})"
                        )
                    recent_tokens = origin_position.end - frame * tokens_per_frame
                    if recent_tokens < 0 or (history_frames != 0 and recent_tokens > origin_position.length):
                        raise ValueError('seed_from source lacks the requested same-time recent history')
                    recent_end = origin_position.length
                    fixed_end = recent_end - recent_tokens if history_frames != 0 else sink_tokens
                    native_sink = min(source_sink_frames * tokens_per_frame, origin_position.end)
                    if fixed_end < sink_tokens or origin_position.length < native_sink:
                        raise ValueError('seed_from source lacks the required sink frames')
                    fixed_start = min(native_sink, fixed_end)
                    spans = [(0, sink_tokens)]
                    if history_frames is None:
                        spans += [(fixed_start, fixed_end), (recent_end, origin_position.length)]
                    elif history_frames:
                        start = max(fixed_start, fixed_end - history_frames * tokens_per_frame)
                        remaining = history_frames * tokens_per_frame - (fixed_end - start)
                        spans += [(start, fixed_end), (recent_end, min(origin_position.length, recent_end + remaining))]
                    length = sum(end - start for start, end in spans)
                    required = length + (recent_tokens if history_frames != 0 else 0)
                    if self.reference_tokens + required > layer['k'].shape[1]:
                        raise ValueError('seed_from target cannot hold fixed history and replacement frames')
                    for field in ('k', 'v', 'k_raw') if 'k_raw' in layer else ('k', 'v'):
                        layer[field][:, :self.reference_tokens].copy_(origin[field][:, :self.reference_tokens])
                        offset = self.reference_tokens
                        for start, end in spans:
                            count = end - start
                            layer[field][:, offset:offset+count].copy_(
                                origin[field][:, self.reference_tokens+start:self.reference_tokens+end])
                            offset += count
                    position.end = frame * tokens_per_frame
                    position.length = length
                    position.rope_base = origin_position.rope_base
                    layer.pop('token_positions', None)
                    layer['sink_tokens'] = sink_frames * tokens_per_frame
                    layer['temporary_reencode'] = True

    def validate_detached(self):
        for layers in self.streams.values():
            for layer in layers:
                if any(torch.is_tensor(value) and (value.requires_grad or value.grad_fn is not None)
                       for value in layer.values()):
                    raise RuntimeError("Committed causal history retains a graph")


def allocate_cache(model, batch_size, capacity_tokens, device, dtype, *, raw_keys=False, sink_tokens=None):
    shape = (batch_size, capacity_tokens, model.num_heads, model.dim // model.num_heads)
    caches = [
        dict(
            k=torch.empty(shape, device=device, dtype=dtype),
            v=torch.empty(shape, device=device, dtype=dtype),
            position=CachePosition(),
        )
        for _ in model.blocks
    ]
    if raw_keys:
        for cache in caches:
            cache["k_raw"] = torch.empty(shape, device=device, dtype=dtype)
    if sink_tokens is not None:
        for cache in caches:
            cache['sink_tokens'] = sink_tokens
    return caches


def cross_attention_cache(model):
    return [dict(is_init=False) for _ in model.blocks]


def resolve_sink_frames(value, native_frames=0):
    """Resolve a per-rollout sink override without mutating the model's defaults."""
    if isinstance(value, str):
        value = int(value)
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError('sink_frames must be a nonnegative integer or null')
    return native_frames if value is None else value


def cache_capacity_frames(frames, window_frames, sink_frames=0, *, block_frames=None):
    """Bound storage by a known rollout length without shrinking reserved sinks.

    ``frames`` includes the temporal prefix; ``window_frames`` includes all
    video sinks. Separate reference-image tokens are added by their backend.
    A -1 window stores the full requested timeline.
    """
    if type(frames) is not int or frames <= 0:
        raise ValueError("Causal cache must contain at least one frame")
    if type(window_frames) is not int or window_frames == 0 or window_frames < -1:
        raise ValueError("Causal cache window must be positive or -1")
    if type(sink_frames) is not int or sink_frames < 0:
        raise ValueError("Causal cache has an invalid attention sink")
    if window_frames != -1 and sink_frames + (block_frames or 0) > window_frames:
        raise ValueError('Causal cache window must fit the sink and current block')
    return max(sink_frames, frames if window_frames == -1 else min(frames, window_frames))


def make_causal_cache(model, batch_size, frames, height, width, device, dtype, *,
                      extra_sink_size=0, capacity_frames=None, sink_frames=None):
    """Allocate model-selected streams under one window and temporal sink policy."""
    if type(extra_sink_size) is not int or extra_sink_size < 0:
        raise ValueError("Extra attention sink size must be non-negative")
    window = -1 if model.local_attn_size == -1 else model.local_attn_size + extra_sink_size
    native_sink = model.sink_size + extra_sink_size
    sink = resolve_sink_frames(sink_frames, native_sink)
    capacity = cache_capacity_frames(frames, window, sink,
                                     block_frames=model.num_frame_per_block)
    if capacity_frames is not None:
        capacity = capacity_frames
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("Causal cache capacity must be a positive frame count")
    if sink_frames is not None and frames > capacity and sink + model.num_frame_per_block > capacity:
        raise ValueError('Causal cache capacity must fit the sink and current block')
    tokens_per_frame = (height // model.patch_size[1]) * (width // model.patch_size[2])
    tokens = capacity * tokens_per_frame
    cache = CausalCache({name: allocate_cache(model, batch_size, tokens, device, dtype,
                                             sink_tokens=None if sink_frames is None else sink * tokens_per_frame)
                         for name in model.cache_streams}, cross_attention_cache(model))
    return cache


def cache_token_positions(cache, sink_tokens=0):
    """Logical temporal token IDs, independent of a pruned cache's storage slots."""
    if 'token_positions' in cache:
        return cache['token_positions']
    cursor = cache['position']
    sink = min(sink_tokens, cursor.length)
    ids = torch.cat((torch.arange(sink, device=cache['k'].device),
                     torch.arange(cursor.end - cursor.length + sink, cursor.end, device=cache['k'].device)))
    return ids.expand(cache['k'].shape[0], -1)


def retained_token_positions(cache, current_start, current_tokens, sink_tokens, rolling_start):
    ids = cache_token_positions(cache, sink_tokens)
    current = torch.arange(current_start, current_start+current_tokens, device=cache['k'].device)
    return torch.cat((ids[:, :sink_tokens], ids[:, rolling_start:], current.expand(ids.shape[0], -1)), 1)


def read_attention(cache, key, value, current_start, sink_tokens, frame_seqlen, *, commit=False, freqs=None,
                   return_positions=False, relative_positions=False):
    """Read or commit native history with its temporal token positions."""
    sink_tokens = cache.get('sink_tokens', sink_tokens)
    current_end = current_start + key.shape[1]
    explicit_positions = return_positions and 'token_positions' in cache
    if commit:
        end = update_cache(cache, key, value, current_start, sink_tokens)
        key, value = cache['k'][:, :end], cache['v'][:, :end]
        positions = cache_token_positions(cache, sink_tokens) if explicit_positions else None
    else:
        view = cache_view(cache, key, value, current_start, sink_tokens, return_positions=explicit_positions)
        key, value = view[:2]
        positions = view[2] if explicit_positions else None
    if not return_positions:
        return key, value, None
    sink = min(sink_tokens, key.shape[1])
    if positions is None:
        ids = (torch.arange(key.shape[1], device=key.device) if relative_positions else
               torch.cat((torch.arange(sink, device=key.device),
                          torch.arange(current_end - key.shape[1] + sink, current_end, device=key.device))))
        positions = ids.expand(key.shape[0], -1)
    return key, value, None, positions


def update_cache(cache, key, value, current_start, sink_tokens=0, *, poses=None):
    """Commit clean KV into the bounded sink and rolling history window."""
    if torch.is_grad_enabled() and (key.requires_grad or value.requires_grad):
        raise RuntimeError("Use cache_view for differentiable KV; commit history under no_grad")
    token_count = key.shape[1]
    current_end = current_start + token_count
    position = cache["position"]
    global_end, local_end = position.end, position.length
    capacity = cache["k"].shape[1]
    ids = cache.get('token_positions')
    sink_tokens = cache.get('sink_tokens', sink_tokens)
    if token_count > capacity:
        raise ValueError("One AR block exceeds the causal cache capacity")
    if sink_tokens < 0 or sink_tokens > capacity:
        raise ValueError("Causal cache has an invalid attention sink")
    if (
        current_start >= sink_tokens
        and sink_tokens
        and token_count > capacity - sink_tokens
    ):
        raise ValueError("One AR block exceeds the rolling cache region")

    if current_end > global_end:
        if current_start != global_end:
            raise ValueError("New causal KV must immediately follow committed history")
        overflow = max(0, local_end + token_count - capacity)
        if overflow:
            rolling_start = min(sink_tokens, local_end)
            rolling_count = local_end - rolling_start
            if overflow > rolling_count:
                raise ValueError("Causal cache overflow would evict its sink")
            remaining = rolling_count - overflow
            cache["k"][:, rolling_start : rolling_start + remaining].copy_(
                cache["k"][:, rolling_start + overflow : local_end].clone()
            )
            cache["v"][:, rolling_start : rolling_start + remaining].copy_(
                cache["v"][:, rolling_start + overflow : local_end].clone()
            )
            local_end = rolling_start + remaining
            if ids is not None:
                ids = torch.cat((ids[:, :rolling_start], ids[:, rolling_start+overflow:]), 1)
        write_start = local_end
        write_end = write_start + token_count
    elif current_end == global_end:
        write_end = local_end
        write_start = write_end - token_count
        if write_start < 0:
            raise ValueError("Repeated denoising has no current KV block")
    else:
        raise ValueError("Causal inference cannot rewrite committed history")

    cache["k"][:, write_start:write_end].copy_(key)
    cache["v"][:, write_start:write_end].copy_(value)
    if ids is not None:
        current_ids = torch.arange(current_start, current_end, device=key.device).expand(key.shape[0], -1)
        cache['token_positions'] = torch.cat((ids[:, :write_start], current_ids), 1)
    position.end = current_end
    position.length = write_end
    return position.length


def cache_view(cache, key, value, current_start, sink_tokens=0, *, return_positions=False):
    """Return detached history plus differentiable current KV."""
    token_count = key.shape[1]
    position = cache["position"]
    global_end, local_end = position.end, position.length
    capacity = cache["k"].shape[1]
    sink_tokens = cache.get('sink_tokens', sink_tokens)
    if current_start != global_end:
        raise ValueError(
            "Read-only causal KV must immediately follow committed history"
        )
    if token_count > capacity:
        raise ValueError("One AR block exceeds the causal cache capacity")
    if sink_tokens < 0 or sink_tokens > capacity:
        raise ValueError("Causal cache has an invalid attention sink")
    sink_count = min(sink_tokens, local_end)
    rolling_capacity = capacity - token_count - sink_count
    if rolling_capacity < 0:
        raise ValueError("One AR block exceeds the rolling cache region")
    rolling_start = max(sink_count, local_end - rolling_capacity)
    result = (
        torch.cat((cache["k"][:, :sink_count].detach(),
                   cache["k"][:, rolling_start:local_end].detach(), key), dim=1),
        torch.cat((cache["v"][:, :sink_count].detach(),
                   cache["v"][:, rolling_start:local_end].detach(), value), dim=1),
    )
    if return_positions:
        return (*result, retained_token_positions(cache, current_start, token_count, sink_count, rolling_start))
    return result

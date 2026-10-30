"""MemRoPE attention and EMA memory compression."""
import torch

from pipeline.causal_inference import CausalInferencePipeline
from .config import validate_memrope_options
from .third_party.memrope.attention import CausalWanSelfAttention as MemAttention, _apply_cache_updates


class MemRoPEAttention(MemAttention):
    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
                kv_cache=None, current_start=0, cache_start=None):
        if kv_cache is None:
            raise RuntimeError("MemRoPE adapter requires cached inference")
        output, update = super().forward(
            x, seq_lens, grid_sizes, freqs, block_mask,
            kv_cache=kv_cache, current_start=current_start, cache_start=cache_start,
        )
        # Apply the cache update after each attention layer.
        _apply_cache_updates(None, [kv_cache], [(0, update)])
        return output


class MemRoPEInferencePipeline(CausalInferencePipeline):
    def inference(self, *args, **kwargs):
        self.kv_cache1 = None
        self.crossattn_cache = None
        return super().inference(*args, **kwargs)

    def _initialize_kv_cache(self, batch_size, dtype, device):
        super()._initialize_kv_cache(batch_size, dtype, device)
        if self.training_free_memory_method == "memrope":
            for cache in self.kv_cache1:
                cache["q"] = torch.zeros_like(cache["k"])
                shape = cache["k"].shape[:2]
                cache["token_temporal_indices"] = torch.zeros(shape, dtype=torch.long, device=device)
                cache["token_spatial_indices"] = torch.zeros(shape, dtype=torch.long, device=device)
                cache["selected_token_indices"] = None
                cache["original_spatial_indices"] = None


def apply_memrope(pipeline, options=None):
    """Configure MemRoPE attention and caching."""
    options = validate_memrope_options(options)
    if not isinstance(pipeline, CausalInferencePipeline) or pipeline.training:
        raise ValueError("MemRoPE requires a dedicated CausalInferencePipeline in eval mode")
    if getattr(pipeline, "training_free_memory_method", None) is not None:
        raise RuntimeError("A memory adapter is already installed on this instance")
    if pipeline.kv_cache1 is not None:
        raise RuntimeError("Apply the adapter before generating any video")
    if pipeline.num_frame_per_block != 3 or pipeline.independent_first_frame:
        raise ValueError("Use 3-frame chunks and independent_first_frame=false")
    attentions = [block.self_attn for block in pipeline.generator.model.blocks]
    if not attentions or any(hasattr(attention, "rope_max_frames") for attention in attentions):
        raise RuntimeError("Unsupported attention configuration")
    before = {name: id(value) for name, value in pipeline.generator.named_parameters()}
    for attention in attentions:
        attention.__class__ = MemRoPEAttention
        attention.local_attn_size = options["cache_frames"]
        attention.sink_size = options["sink_frames"]
        attention.recent_size = options["recent_frames"]
        attention.max_attention_size = options["cache_frames"] * pipeline.frame_seq_length
        for name in ("compression_method", "ema_alpha_long", "ema_alpha_short", "ema_adaptive", "use_block_rope"):
            setattr(attention, name, options[name])
    pipeline.local_attn_size = options["cache_frames"]
    pipeline.generator.model.local_attn_size = options["cache_frames"]
    pipeline.training_free_memory_method = "memrope"
    pipeline.training_free_memory_options = dict(options)
    pipeline.__class__ = MemRoPEInferencePipeline
    if before != {name: id(value) for name, value in pipeline.generator.named_parameters()}:
        raise RuntimeError("Applying MemRoPE unexpectedly replaced model parameters")
    return dict(pipeline.training_free_memory_options)

from wan.modules.attention import attention
from utils.attention import validate_local_attn_size
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
import os

# Use max-autotune for Wan 1.3B's attention head dimensions.
# See https://github.com/pytorch/pytorch/issues/133254.
flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode=os.environ.get("WAN_FLEX_COMPILE_MODE", "default")
)



def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
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
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
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
                 local_attn_size=21,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = validate_local_attn_size(local_attn_size)
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = self.local_attn_size * 1560

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
        cache_start=None
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
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=21,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = validate_local_attn_size(local_attn_size)
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=21,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to 21):
                Positive temporal attention window in latent frames, including the current block
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = validate_local_attn_size(local_attn_size)
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None
        self.block_mask_signature = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(
        self, module=None, value=False, enable=None, gradient_checkpointing_func=None
    ):
        # Support Diffusers' legacy module/value and current enable interfaces.
        self.gradient_checkpointing = value if enable is None else enable

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=21
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        local_attn_size = validate_local_attn_size(local_attn_size)
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)


        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1,
        local_attn_size: int = 21, sink_size: int = 0,
    ) -> BlockMask:
        """Build a teacher-forcing mask matching the inference KV cache.

        The input sequence is ``[clean context][noisy targets]``.  A noisy block
        sees its own noisy tokens and only the clean history that would remain in
        the rolling inference cache.  For the long-video AR teacher,
        ``local_attn_size=21``, ``sink_size=3`` and a three-frame block produce
        exactly ``sink3 + recent15 + current3``. Set the window to at least the
        complete timeline length to use all preceding clean frames.
        """
        local_attn_size = validate_local_attn_size(local_attn_size)
        if num_frames <= 0 or frame_seqlen <= 0 or num_frame_per_block <= 0:
            raise ValueError(
                "num_frames, frame_seqlen and num_frame_per_block must be positive")
        if num_frames % num_frame_per_block != 0:
            raise ValueError(
                f"num_frames={num_frames} must be divisible by "
                f"num_frame_per_block={num_frame_per_block}")
        if sink_size < 0:
            raise ValueError(f"sink_size must be non-negative, got {sink_size}")
        minimum_cache = sink_size + num_frame_per_block
        if local_attn_size < minimum_cache:
            raise ValueError(
                "local_attn_size must fit sink plus the current block: "
                f"{local_attn_size} < {sink_size}+{num_frame_per_block}")

        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        mask_length = total_length + padded_length
        context_recent_starts = torch.zeros(
            mask_length, device=device, dtype=torch.long)
        context_ends = torch.zeros(mask_length, device=device, dtype=torch.long)
        noise_context_recent_starts = torch.zeros(
            mask_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(mask_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(mask_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(mask_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        sink_tokens = sink_size * frame_seqlen
        # The noisy half holds the current block separately, leaving one fewer
        # clean block in its recent-history budget than a clean cache commit.
        clean_recent_capacity = (local_attn_size - sink_size) * frame_seqlen
        noisy_recent_capacity = (
            local_attn_size - sink_size - num_frame_per_block
        ) * frame_seqlen

        # A clean block sees sink + rolling recent + itself, matching a cache commit.
        for start in frame_indices:
            start_i = int(start.item())
            end_i = min(start_i + attention_block_size, clean_ends)
            recent_start = max(sink_tokens, end_i - clean_recent_capacity)
            context_recent_starts[start_i:end_i] = recent_start
            context_ends[start_i:end_i] = end_i

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # A noisy block sees clean sink + recent history and its own noisy block;
        # the clean version of the current block is deliberately excluded.
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            history_end = block_index * attention_block_size
            recent_start = max(sink_tokens, history_end - noisy_recent_capacity)
            noise_context_recent_starts[start:end] = recent_start
            noise_context_ends[start:end] = history_end

        def attention_mask(b, h, q_idx, kv_idx):
            clean_history = (kv_idx < context_ends[q_idx]) & (
                (kv_idx < sink_tokens) |
                (kv_idx >= context_recent_starts[q_idx])
            )
            clean_mask = (q_idx < clean_ends) & clean_history

            same_noisy_block = (
                (kv_idx < noise_noise_ends[q_idx]) &
                (kv_idx >= noise_noise_starts[q_idx])
            )
            clean_history_for_noisy = (kv_idx < noise_context_ends[q_idx]) & (
                (kv_idx < sink_tokens) |
                (kv_idx >= noise_context_recent_starts[q_idx])
            )
            noise_mask = (q_idx >= clean_ends) & (
                same_noisy_block | clean_history_for_noisy)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(
            attention_mask, B=None, H=None, Q_LEN=mask_length,
            KV_LEN=mask_length, _compile=True, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=21
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        local_attn_size = validate_local_attn_size(local_attn_size)
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)


        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Process the current latent chunk using cached attention history.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.

        Args:
            x (Tensor or List[Tensor]):
                Input video latents [B, C_in, F, H, W] or a list of [C_in, F, H, W] tensors
            t (Tensor):
                Per-frame diffusion timesteps with shape [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum latent-token sequence length
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Image conditioning latents aligned with x in time and space; concatenated along channels

        Returns:
            Tensor:
                Batched latent-space predictions [B, C_out, F, H, W], matching the input latent resolution
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module, kv_cache_snapshot=None):
            """Keep checkpoint recomputation independent of the rolling KV cache.

            The first invocation must update the live cache for the autoregressive
            rollout.  PyTorch invokes this closure again during backward, after the
            live cache has already advanced; recomputation therefore needs the cache
            state from immediately before the original forward.  Store that state on
            CPU so the snapshots do not consume the GPU memory saved by checkpointing.
            """
            first_invocation = True

            def custom_forward(*inputs, **call_kwargs):
                nonlocal first_invocation
                if first_invocation or kv_cache_snapshot is None:
                    first_invocation = False
                    return module(*inputs, **call_kwargs)

                device = inputs[0].device
                recompute_cache = {
                    name: value.to(device=device, non_blocking=False)
                    for name, value in kv_cache_snapshot.items()
                }
                recompute_kwargs = dict(call_kwargs)
                recompute_kwargs["kv_cache"] = recompute_cache
                return module(*inputs, **recompute_kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_kwargs = dict(kwargs)
                block_kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                kv_cache_snapshot = {
                    name: value.detach().to(device="cpu", copy=True)
                    for name, value in kv_cache[block_index].items()
                }
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, kv_cache_snapshot),
                    x, **block_kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor or List[Tensor]):
                Input video latents [B, C_in, F, H, W] or a list of [C_in, F, H, W] tensors
            t (Tensor):
                Per-frame diffusion timesteps with shape [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum latent-token sequence length
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Image conditioning latents aligned with x in time and space; concatenated along channels

        Returns:
            Tensor:
                Batched latent-space predictions [B, C_out, F, H, W], matching the input latent resolution
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Cache blockwise causal masks by temporal and spatial shape.
        mask_signature = (
            clean_x is not None,
            int(x.shape[2]),
            int(x.shape[-2]),
            int(x.shape[-1]),
            int(self.num_frame_per_block),
            int(self.local_attn_size),
            int(self.sink_size),
        )
        if self.block_mask is None or self.block_mask_signature != mask_signature:
            if clean_x is not None:  # Teacher forcing with clean context.
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                        sink_size=self.sink_size,
                    )
            else:  # Causal attention over noisy blocks.
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
            self.block_mask_signature = mask_signature

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    @torch.no_grad()
    def _forward_history(self, x, t, context, seq_len, history_request):
        """Run both original cache trajectories within one FSDP forward."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # Local early return is safe only after entering a whole-model FSDP unit,
        # with no nested parameter collectives left inside the cache trajectories.
        if any(isinstance(module, FSDP) for module in self.modules()):
            raise RuntimeError("History fusion requires a whole-model FSDP unit without nested FSDP")
        if not history_request["active_mask"].any().item():
            return torch.zeros_like(x), torch.zeros_like(x)
        if history_request["num_layers"] != len(self.blocks):
            raise ValueError("AR history cache layer count differs from the teacher")

        batch, _, frames = x.shape[:3]
        block = history_request["block_size"]
        seq = history_request["frame_seq_length"]
        dtype = history_request["cache_dtype"]
        heads, head_dim = self.num_heads, self.dim // self.num_heads
        outputs = []
        for branch in history_request["branches"]:
            sink, recent = branch["sink"], branch["recent"]
            capacity = min(self.local_attn_size, sink.shape[1] + recent.shape[1] + frames)
            start = branch["global_start"] * seq
            kv = [{
                "k": torch.zeros([batch, capacity * seq, heads, head_dim], dtype=dtype, device=x.device),
                "v": torch.zeros([batch, capacity * seq, heads, head_dim], dtype=dtype, device=x.device),
                "global_end_index": torch.tensor([start], dtype=torch.long, device=x.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=x.device),
            } for _ in self.blocks]
            cross = [{
                "k": torch.zeros([batch, self.text_len, heads, head_dim], dtype=dtype, device=x.device),
                "v": torch.zeros([batch, self.text_len, heads, head_dim], dtype=dtype, device=x.device),
                "is_init": False,
            } for _ in self.blocks]
            for part, (source, base) in enumerate(((sink, 0), (recent, branch["recent_start"]))):
                if part == 1 and sink.shape[1]:
                    for index in range(len(kv)):
                        kv[index]["global_end_index"].fill_(branch["recent_start"] * seq)
                for offset in range(0, source.shape[1], block):
                    clean = source[:, offset:offset + block]
                    self._forward_inference(
                        clean.permute(0, 2, 1, 3, 4),
                        t=torch.zeros([batch, clean.shape[1]], device=x.device, dtype=torch.int64),
                        context=context, seq_len=seq_len,
                        kv_cache=kv, crossattn_cache=cross,
                        current_start=(base + offset) * seq,
                    )
            outputs.append(self._forward_inference(
                x, t=t, context=context, seq_len=seq_len,
                kv_cache=kv, crossattn_cache=cross,
                current_start=history_request["chunk_position"] * seq,
            ))
            # The positive cache is no longer needed after its query. Release it
            # before allocating the negative cache while weights remain gathered.
            del kv, cross
        return tuple(outputs)

    def forward(
        self,
        *args,
        **kwargs
    ):
        history_request = kwargs.pop('history_request', None)
        if history_request is not None:
            return self._forward_history(*args, history_request=history_request, **kwargs)
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # Parallel forward with teacher-forcing or causal masks.
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor or List[Tensor]):
                Patch features [B, L, C_out * prod(patch_size)] or a list of per-sample tensors
            grid_sizes (Tensor):
                Spatiotemporal patch-grid dimensions,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed latent tensors with shape [C_out, F, H, W]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

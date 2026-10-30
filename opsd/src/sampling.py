"""Shared full-context and autoregressive sampling primitives."""

import torch
from tqdm import tqdm
from scheduler import sampling_step

@torch.no_grad()
def sample_full_context(
    backend, noise, context, negative_context, *, condition,
    steps=None, shift=None, guide_scale=None, offload_model=False,
    sampling_mode="ode", noise_generator=None, **conditioning,
):
    """Generate [B,C,F,H,W]; camera and alternative conditions flow to predict."""
    from utils import text_cfg

    if noise.ndim != 5:
        raise ValueError('Noise must have shape [B,C,F,H,W]')
    scheduler = backend.sampling_scheduler(
        noise.device,
        backend.config.sample_steps if steps is None else steps,
        backend.sampling_shift(noise) if shift is None else shift,
        sampling_mode,
    )
    guide_scale = backend.config.sample_guide_scale if guide_scale is None else guide_scale
    patch = backend.config.patch_size
    if any(size % p for size, p in zip(noise.shape[2:], patch, strict=True)):
        raise ValueError('Latent shape must align with the patch grid')
    seq_len = noise.shape[2] * noise.shape[3] * noise.shape[4] // (patch[0] * patch[1] * patch[2])
    if any(getattr(model, 'sequence_parallel', False) for model in backend.models.values()):
        from wan.distributed.util import get_world_size
        world_size = get_world_size()
        seq_len = (seq_len + world_size - 1) // world_size * world_size
    latent = condition.restore(noise)
    active = None
    try:
        for index, timestep in enumerate(scheduler.timesteps):
            model, scale = backend.select_model(timestep, guide_scale)
            if model is not active:
                active = model
                if offload_model:
                    for inactive in backend.models.values():
                        if inactive is not model:
                            inactive.cpu()
                model.to(noise.device)
            t = backend.model_timesteps(timestep, latent, condition, seq_len)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                positive = (backend.predict(model, latent, t, context, seq_len, condition, **conditioning)
                            if scale != 0.0 else None)
                negative = (backend.predict(model, latent, t, negative_context, seq_len, condition, **conditioning)
                            if scale != 1.0 else None)
                flow = text_cfg(positive, negative, scale)
                # Official full-context sampling also runs UniPC inside autocast.
                transition_noise = (torch.randn(latent.shape, generator=noise_generator,
                                               device=latent.device, dtype=latent.dtype) if scheduler.stochastic_sampling
                                    and index + 1 < len(scheduler.timesteps) else None)
                latent = condition.restore(sampling_step(scheduler, flow, latent, index, noise=transition_noise))
    finally:
        if offload_model and active is not None:
            active.cpu()
    return latent

def denoise_block(latent, timesteps, predict_flow, step):
    """Shared denoising loop; the caller owns model prediction and solver update."""
    for index, timestep in enumerate(timesteps):
        flow = predict_flow(latent, timestep)
        latent = step(flow, timestep, latent, index)
    return latent

def denoise_few_step_block(latent, scheduler, predict_flow, draw_noise, *, restore=None):
    """Denoise one AR block with the selected ODE/SDE update.

    The caller supplies flow prediction and transition noise; the final step
    returns clean latents without another noise draw. ``restore`` reapplies any
    known image condition after each update, including the final clean step.
    """
    scheduler.reset()

    def step(flow, timestep, current, index):
        noise = (draw_noise(current) if scheduler.stochastic_sampling
                 and index + 1 < len(scheduler.timesteps) else None)
        updated = sampling_step(scheduler, flow, current, index, noise=noise)
        return updated if restore is None else restore(updated)

    return denoise_block(latent, scheduler.timesteps, predict_flow, step)

@torch.no_grad()
def sample_causal_sequence(
    noise, block_lengths, prepare_block, denoise, *, prefix=None,
    prefix_lengths=(), include_prefix=False, progress=None,
):
    """Prefill clean history and generate blocks, with time on tensor axis 1.

    ``noise`` contains only positions to generate, or is a ``noise(length)``
    factory drawing one block at a time. A factory requires a clean prefix as
    the output's shape/device/dtype reference, but does not prefill extra frames.
    ``prepare_block(start, length)``
    uses absolute latent-frame indices, including the prefilled prefix, and returns
    ``(predict_flow, commit_clean)`` callbacks bound to that block. The backend owns
    conditioning, cache semantics and RNG; ``denoise(noise, predict_flow)`` owns the
    solver. Prefix blocks only commit; generated blocks denoise then commit once.

    The remaining axes keep their native order, preserving official random draws.
    Output uses the noise dtype/device and optionally includes the clean prefix.
    """
    block_lengths, prefix_lengths = tuple(block_lengths), tuple(prefix_lengths)
    draw_block = noise if callable(noise) else None
    if draw_block is not None:
        if prefix is None or prefix.ndim < 2:
            raise ValueError('A block noise factory requires a clean prefix tensor')
        prototype = prefix
        generated_frames = sum(block_lengths)
    else:
        if noise.ndim < 2 or noise.shape[1] <= 0:
            raise ValueError('Noise must have a nonempty temporal axis 1')
        prototype = noise
        generated_frames = noise.shape[1]
    if prefix is not None and prefix.ndim != prototype.ndim:
        raise ValueError('Prefix and noise must have the same number of axes')
    prefix_frames = 0 if prefix is None else prefix.shape[1]
    for lengths, frames in ((block_lengths, generated_frames), (prefix_lengths, prefix_frames)):
        if any(type(length) is not int or length <= 0 for length in lengths) or sum(lengths) != frames:
            raise ValueError('Positive integer block lengths must cover the complete timeline')
    if not generated_frames:
        raise ValueError('Noise must have a nonempty temporal axis 1')
    if prefix is not None:
        if prefix.shape[:1] + prefix.shape[2:] != prototype.shape[:1] + prototype.shape[2:]:
            raise ValueError('Prefix and noise must agree on all non-temporal axes')
        if prefix.device != prototype.device:
            raise ValueError('Prefix and noise must be on the same device')
    offset = prefix_frames if include_prefix else 0
    shape = list(prototype.shape)
    shape[1] = generated_frames + offset
    output = prototype.new_empty(shape)
    if offset:
        output[:, :offset] = prefix
    start = 0
    for length in prefix_lengths:
        _, commit = prepare_block(start, length)
        commit(prefix[:, start:start + length])
        start += length
    lengths = tqdm(block_lengths, desc=progress) if progress is not None else block_lengths
    start = 0
    for length in lengths:
        stop = start + length
        if draw_block is not None:
            current = draw_block(length)
            expected = (prototype.shape[0], length, *prototype.shape[2:])
            if (current.shape != expected or current.device != prototype.device
                    or current.dtype != prototype.dtype):
                raise ValueError('Block noise must match the prefix shape, device and dtype')
        predict, commit = prepare_block(prefix_frames + start, length)
        if draw_block is None:
            current = noise[:, start:stop]
        clean = denoise(current, predict)
        if clean.shape != current.shape:
            raise ValueError('Denoising must preserve the current block shape')
        commit(clean)
        output[:, offset + start:offset + stop] = clean
        del clean
        start = stop
    return output

# Image conditioning follows WorldKV under Apache-2.0; see repository LICENSE.
def encode_video_condition(codec, video, frame_num, vae_stride=(4, 8, 8)):
    """Encode [B,3,P,H,W] prefix + zero RGB and prepend native temporal masks."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError('Condition video must have shape [B,3,P,H,W]')
    prefix = video.shape[2]
    temporal, height_stride, width_stride = vae_stride
    if not 0 < prefix < frame_num or (prefix - 1) % temporal or (frame_num - 1) % temporal:
        raise ValueError('RGB prefix and output must align with the causal VAE timeline')
    if video.shape[3] % height_stride or video.shape[4] % width_stride:
        raise ValueError('Resize RGB and transform camera intrinsics before encoding')
    padded = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, frame_num - prefix))
    encoded = torch.stack(codec.encode(list(padded)))
    mask = video.new_zeros(video.shape[0], 1, frame_num, *encoded.shape[-2:])
    mask[:, :, :prefix] = 1
    mask = torch.cat((mask[:, :, :1].repeat_interleave(temporal, dim=2), mask[:, :, 1:]), dim=2)
    mask = mask.unflatten(2, (-1, temporal)).transpose(2, 3).flatten(1, 2)
    return torch.cat((mask, encoded), dim=1)

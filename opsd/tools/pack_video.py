#!/usr/bin/env python3
"""Pack an image/video manifest or image/control Parquet for inference, SFT or OPSD."""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from backbones import add_model_arg, load_backend
from dataset import LatentSequenceDataset, VideoDataset, packing_identity, read_packing_source, write_latent_lmdb
from engine.runtime import distributed_phase
from utils import normalize_pixels


@torch.no_grad()
def encoded_samples(dataset, backend, codec, device, *, visual_encoder=None, start=0):
    """Encode every visual input through the selected backend, once per clip."""
    dtype = torch.bfloat16
    keep_pixels = dataset.keep_condition_pixels
    channels, latent_h, latent_w = backend.latent_shape
    stride_h, stride_w = backend.pixel_shape[0] // latent_h, backend.pixel_shape[1] // latent_w

    def encode(pixels):
        value = backend.encode(codec, normalize_pixels(pixels).to(dtype), device, dtype).cpu()
        if value.ndim != 4 or not torch.isfinite(value).all():
            raise ValueError("Backend encode must return finite [F,C,H,W] latents")
        return value

    for index in range(start, len(dataset)):
        raw = dataset[index]
        frames, prefix = raw['num_latent_frames'], raw['condition_frames']
        size = raw['size']
        if size[0] % stride_h or size[1] % stride_w:
            raise ValueError("Output size must be divisible by the codec spatial stride")
        shape = (channels, size[0] // stride_h, size[1] // stride_w)
        sample = {name: value for name, value in raw.items() if name not in (
            'pixels', 'reference_pixels', 'num_latent_frames', 'condition_frames', 'size', 'has_target')}
        if not keep_pixels:
            sample.pop('condition_pixels', None)
        sample.update(target_frames=frames-prefix, latent_frame_shape=shape)
        pixels, prefix_rgb = raw['pixels'], None
        if pixels is not None:
            print(f"Encode {index + 1}/{len(dataset)}: {pixels.shape[1]} RGB frames, {size}", flush=True)
            latent = encode(pixels)
            expected = frames if raw['has_target'] else prefix
            if tuple(latent.shape) != (expected, *shape):
                raise ValueError(f"Encoded shape {tuple(latent.shape)} does not match {(expected, *shape)}")
            sample['condition_latent'] = latent[:prefix]
            if raw['has_target']:
                sample['clean_latent'] = latent[prefix:]
            if prefix:
                prefix_rgb = pixels[:, :backend.decoded_frames(prefix)]
        else:
            sample['condition_latent'] = torch.empty((0, *shape), dtype=dtype)
        if prefix_rgb is not None and backend.requires_visual_condition_encoding:
            features = backend.encode_visual_condition(
                codec, normalize_pixels(prefix_rgb).to(device=device, dtype=dtype)[None],
                frames, encoder=visual_encoder)
            sample.update({name: value.cpu() for name, value in features.items()})
        if 'reference_pixels' in raw:
            refs = [encode(rgb[:, None]).permute(1, 0, 2, 3) for rgb in raw['reference_pixels']]
            if any(value.shape[0] != channels for value in refs):
                raise ValueError("Reference and video codecs must share latent channels")
            sample.update(reference_latent=torch.stack(refs), reference_mask=torch.ones(len(refs), dtype=torch.uint8))
        del raw, pixels, prefix_rgb
        yield sample


def _merged_samples(paths, start):
    offset = 0
    for path in paths:
        with LatentSequenceDataset(path) as dataset:
            dataset.load_latent = dataset.has_target
            for index in range(max(0, start-offset), len(dataset)):
                sample = dataset[index]
                if dataset.has_prompt_switch and 'teacher_prompt_mask' not in sample:
                    sample['teacher_prompt_mask'] = torch.zeros(sample['target_frames'], dtype=torch.bool)
                yield sample
            offset += len(dataset)


def pack_dataset(dataset, backend, model_dir, output, *, device=None):
    """Media uses CUDA by default; torchrun writes disjoint contiguous row shards."""
    output = Path(output).resolve()
    identity = packing_identity(dataset, backend, model_dir)
    if (output / 'data.mdb').exists():
        write_latent_lmdb(output, (), identity=identity)
        return
    world_size, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))

    def select_device(needs_encoding):
        if device is not None:
            return torch.device(device)
        if not needs_encoding:
            return torch.device('cpu')
        if not torch.cuda.is_available():
            raise RuntimeError("VAE packing requires CUDA")
        selected = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        torch.cuda.set_device(selected)
        return selected

    def write(rows, destination, signature):
        def samples(start):
            if start == len(rows):
                return
            needs_encoding = rows.requires_encoding(start)
            selected = select_device(needs_encoding)
            codec = encoder = None
            if needs_encoding:
                codec = backend.codec(model_dir, selected)
                encoder = backend.visual_encoder(model_dir, selected)
            yield from encoded_samples(rows, backend, codec, selected, visual_encoder=encoder, start=start)
        write_latent_lmdb(destination, samples, identity=signature)

    if world_size == 1:
        write(dataset, output, identity)
        return
    device = select_device(dataset.requires_encoding())
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(hours=4),
                            device_id=device if device.type == 'cuda' else None)
    shards = output.with_name('.' + output.name + '.shards')
    try:
        with distributed_phase(None, 'Packing (completed rows retained)'):
            start, stop = len(dataset) * rank // world_size, len(dataset) * (rank+1) // world_size
            if start < stop:
                local = VideoDataset(dataset.rows[start:stop], dataset.root,
                                     pixel_shape=dataset.pixel_shape, decoded_frames=dataset.decoded_frames)
                local.keep_condition_pixels = dataset.keep_condition_pixels
                write(local, shards / str(rank), {'pack': identity, 'span': [start, stop]})
        with distributed_phase(None, 'Packing shard merge'):
            if rank == 0:
                paths = [shards / str(i) for i in range(world_size)
                         if len(dataset) * i // world_size < len(dataset) * (i+1) // world_size]
                write_latent_lmdb(output, lambda start: _merged_samples(paths, start), identity=identity)
                shutil.rmtree(shards)
    finally:
        dist.destroy_process_group()


def pack_replacing_existing(dataset, backend, model_dir, output):
    """Publish a new LMDB before replacing an older pack at the same path."""
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError(f"Refusing to replace a symlink: {output}")
    output = output.resolve()
    if not output.exists():
        pack_dataset(dataset, backend, model_dir, output)
        return
    if not output.is_dir() or not (output / 'data.mdb').is_file():
        raise ValueError(f"Refusing to replace a directory without a packed LMDB: {output}")
    try:
        write_latent_lmdb(output, (), identity=packing_identity(dataset, backend, model_dir))
    except FileExistsError:
        pass
    else:
        return  # The current pack already matches the manifest and model assets.

    parent = output.parent.resolve()
    token = uuid4().hex
    replacement = parent / f'.{output.name}.replacement-{token}'
    backup = parent / f'.{output.name}.backup-{token}'

    def remove_staging(path):
        if path.exists():
            if path.is_symlink() or path.resolve().parent != parent:
                raise ValueError(f"Refusing to remove a staging path outside {parent}: {path}")
            shutil.rmtree(path)

    try:
        pack_dataset(dataset, backend, model_dir, replacement)
        try:
            output.rename(backup)
            replacement.rename(output)
        except BaseException:
            if backup.exists() and not output.exists():
                backup.rename(output)
            raise
        remove_staging(backup)
        print(f"Replaced old pack: {output}", flush=True)
    finally:
        remove_staging(replacement)
        remove_staging(replacement.with_name('.' + replacement.name + '.building'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON/JSONL manifest or self-contained image/control Parquet")
    add_model_arg(parser)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true",
                        help="Rebuild a stale LMDB and replace it only after the new pack succeeds")
    parser.add_argument("--num-latent-frames", type=int, default=None)
    args = parser.parse_args()
    backend = load_backend(args.model)
    if args.manifest.suffix == '.parquet' and args.num_latent_frames is not None:
        parser.error('--num-latent-frames cannot override a Parquet action timeline')
    rows, root = read_packing_source(args.manifest, backend, args.output_dir)
    if args.num_latent_frames is not None:
        for row in rows:
            row["num_latent_frames"] = args.num_latent_frames
    dataset = VideoDataset(rows, root, pixel_shape=backend.pixel_shape,
                           decoded_frames=backend.decoded_frames)
    if args.replace_existing:
        if int(os.environ.get('WORLD_SIZE', 1)) != 1:
            parser.error('--replace-existing requires a single packing process')
        pack_replacing_existing(dataset, backend, args.model_dir, args.output_dir)
    else:
        pack_dataset(dataset, backend, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()

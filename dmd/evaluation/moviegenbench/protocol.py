"""Fixed MovieGenBench 60s generation protocol, one video per source row."""
import hashlib
import json
from pathlib import Path

from evaluation.generator import write_json
from evaluation.video import valid_video as _valid_video


PROTOCOL = 'moviegenbench-60s-vbenchlong-v1'
FPS, LATENT_FRAMES = 16, 252
PIXEL_FRAMES = 4 * LATENT_FRAMES - 3


def sample_seed(source_index):
    # Keep the established per-row RNG stream while fixing one sample per row.
    value = f'moviegen100-context-window-v1:{source_index}:0'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'big')


def episode_name(row):
    # VBench-Long splits at underscores and recognizes cached folders by -N.
    return f"p{row['source_index']:04d}-0"


def valid_video(path):
    return _valid_video(path, PIXEL_FRAMES, FPS)


def prepare(source, output):
    source = Path(source)
    rows = [{'source_index': index, 'prompt': text.strip()}
            for index, text in enumerate(source.read_text(encoding='utf-8').splitlines())
            if text.strip()]
    if not rows:
        raise ValueError('MovieGenBench source contains no prompts')
    manifest = {'protocol': PROTOCOL,
                'source_nonempty_prompts': len(rows),
                'source_unique_prompts': len({row['prompt'] for row in rows}),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'prompts': rows, 'samples_per_prompt': 1,
                'fps': FPS, 'latent_frames': LATENT_FRAMES, 'pixel_frames': PIXEL_FRAMES}
    target = Path(output) / 'prompt_manifest.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != manifest:
            raise ValueError('MovieGenBench protocol or source changed; choose another --output-dir')
    else:
        write_json(target, manifest)
    return manifest


def generation_receipt(directory, row):
    directory = Path(directory)
    name = episode_name(row)
    video = directory / 'videos' / (name + '.mp4')
    target = directory / 'generation' / (name + '.json')
    try:
        receipt = json.loads(target.read_text(encoding='utf-8'))
        if (receipt['protocol'] != PROTOCOL or receipt['source_index'] != row['source_index']
                or receipt['prompt'] != row['prompt'] or receipt['seed'] != sample_seed(row['source_index'])
                or not valid_video(video) or video.stat().st_mtime_ns > target.stat().st_mtime_ns):
            return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None

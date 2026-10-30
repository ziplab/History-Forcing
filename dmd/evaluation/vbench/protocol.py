"""Fixed 15s VBench quality and semantic protocols with official prompt labels."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SAMPLES_PER_PROMPT = 5
LATENT_FRAMES = 63
PIXEL_FRAMES = 249
FPS = 16
BATCH_SIZE = 4
SEED = 0
SUITES = {
    'quality': {
        'subject_consistency': ('subject_consistency', 'dynamic_degree', 'motion_smoothness'),
        'temporal_flickering': ('temporal_flickering',),
        'scene': ('background_consistency',),
        'overall_consistency': ('aesthetic_quality', 'imaging_quality'),
    },
    'semantic': {name: (name,) for name in (
        'object_class', 'multiple_objects', 'human_action', 'color',
        'spatial_relationship', 'scene', 'appearance_style', 'temporal_style',
        'overall_consistency',
    )},
}
# Keep the original eleven-group TXT order and verify every shared label set.
GROUPS = dict(SUITES['quality'])
for _group, _dimensions in SUITES['semantic'].items():
    GROUPS[_group] = tuple(dict.fromkeys((*GROUPS.get(_group, ()), *_dimensions)))
PROMPT_COUNTS = {
    'subject_consistency': 72, 'temporal_flickering': 75,
    'scene': 86, 'overall_consistency': 93, 'object_class': 79,
    'multiple_objects': 82, 'human_action': 100, 'color': 85,
    'spatial_relationship': 84, 'appearance_style': 90, 'temporal_style': 100,
}
PROMPT_LABEL_SHA256 = {
    'subject_consistency': 'b6b72ab4799acd7fa9bee99c34b938df0c49ca303a0abf38f17ebe15c8085c49',
    'temporal_flickering': '9dac1e50d27f8a44f9329113183db36b61dfc41393b86fbf3ec5e07391f0cda6',
    'scene': '87572cdb4b2d3d15bedd63edc5ac0326999fc6182d28494e02d11d41579f0bd5',
    'overall_consistency': '98989092f921763a33517fd11105702a7fe63dacee6a260bb260d9759da4e29f',
    'object_class': '988195c4176126e3c7ea8db6db554237a76b322b32c75d694916085424e4d293',
    'multiple_objects': '6ccd6359671b5e421a417c1ba454781abd55c857ef54884e7b4e694e1986376d',
    'human_action': '15442d4859ec73a181f9ff48680d3175a7ad59ae009ab04d5284975a4671cc09',
    'color': 'ea9cf41ba01c21a94e6a95c91478a74d9862765f7680009d102aefbe15963daa',
    'spatial_relationship': 'aa22a46ea68472d6319f30cd620722e7b23f3e3a1ddf99a76debb8498cd553da',
    'appearance_style': '78f0fb80c9a27c60dc4b681ca4d2e608b6697f9d9aaf903a63b987d118c9954e',
    'temporal_style': 'de77b31e5c7ed2112d4a859b5746ea066028d7de886ed3bbcde99553e59d0766',
}


def resolve_dimensions(dimensions=None, *, suite='quality'):
    """Validate a CSV/list selection and return the suite's canonical order."""
    if suite not in SUITES:
        raise ValueError(f'Unknown VBench suite: {suite}')
    ordered = tuple(dimension for names in SUITES[suite].values() for dimension in names)
    if dimensions is None:
        return ordered
    if isinstance(dimensions, str):
        selected = [name.strip() for name in dimensions.split(',')]
    elif isinstance(dimensions, (list, tuple)):
        selected = [name.strip() if isinstance(name, str) else name for name in dimensions]
    else:
        raise ValueError('VBench dimensions must be a CSV string or list')
    if not selected or any(not isinstance(name, str) or not name for name in selected):
        raise ValueError('VBench dimensions must contain nonempty metric names')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate VBench dimensions are not allowed')
    unknown = set(selected) - set(ordered)
    if unknown:
        raise ValueError(f'Unknown VBench {suite} dimensions: {sorted(unknown)}')
    return tuple(name for name in ordered if name in selected)


def selected_groups(dimensions=None, *, suite='quality'):
    selected = set(resolve_dimensions(dimensions, suite=suite))
    return {group: tuple(name for name in names if name in selected)
            for group, names in SUITES[suite].items() if selected.intersection(names)}


def selection_id(dimensions=None, *, suite='quality'):
    selected = resolve_dimensions(dimensions, suite=suite)
    if selected == resolve_dimensions(suite=suite):
        return None
    digest = hashlib.sha256(','.join(selected).encode('utf-8')).hexdigest()[:12]
    return f'{suite}{len(selected)}_{digest}'


def read_prompts(prompt_file, group, full_info):
    """Pair one fixed TXT block with labels from official VBench metadata.

    The TXT contains Wan seed42 expansions in the eleven GROUPS' original order.
    Official metadata provides the short prompts used for standard video names.
    """
    if group not in GROUPS:
        raise ValueError(f'Unknown VBench prompt group: {group}')
    expanded = Path(prompt_file).read_text(encoding='utf-8').splitlines()
    if len(expanded) != sum(PROMPT_COUNTS.values()) or not all(p.strip() for p in expanded):
        raise ValueError('VBench prompt TXT must contain exactly 946 nonempty lines')
    metadata = json.loads(Path(full_info).read_text(encoding='utf-8'))
    prompts = [row['prompt_en'] for row in metadata if GROUPS[group][0] in row['dimension']]
    if len(prompts) != PROMPT_COUNTS[group] or len(set(prompts)) != len(prompts):
        raise ValueError(f'Invalid official prompt list: {group}')
    if hashlib.sha256('\n'.join(prompts).encode('utf-8')).hexdigest() != PROMPT_LABEL_SHA256[group]:
        raise ValueError(f'Official prompt labels or order changed: {group}')
    for dimension in GROUPS[group]:
        expected = [row['prompt_en'] for row in metadata if dimension in row['dimension']]
        if expected != prompts:
            raise ValueError(f'Official prompt order differs within group: {dimension}')
    if any(not p.strip() or '/' in p or '\\' in p
           or len((p + '-0.mp4').encode()) > 255 for p in prompts):
        raise ValueError(f'Prompt cannot be used as a video filename: {group}')
    start = 0
    for name in GROUPS:
        if name == group:
            break
        start += PROMPT_COUNTS[name]
    return prompts, expanded[start:start + PROMPT_COUNTS[group]]


def build_plan(prompt_file, full_info, dimensions=None, *, suite='quality'):
    plan = []
    for group, names in selected_groups(dimensions, suite=suite).items():
        prompts, _ = read_prompts(prompt_file, group, full_info)
        plan.append((group, len(prompts) * SAMPLES_PER_PROMPT, names))
    return plan


def expected_videos(prompt_file, group, directory, full_info):
    prompts, _ = read_prompts(prompt_file, group, full_info)
    return [Path(directory) / f'{prompt}-{sample}.mp4'
            for prompt in prompts for sample in range(SAMPLES_PER_PROMPT)]


def missing_videos(prompt_file, group, directory, full_info):
    from evaluation.video import valid_video
    return [path for path in expected_videos(prompt_file, group, directory, full_info)
            if not valid_video(path, PIXEL_FRAMES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'validate-videos', 'missing-videos'))
    parser.add_argument('--suite', choices=SUITES, default='quality')
    parser.add_argument('--dimensions', help='Optional comma-separated metric subset within --suite.')
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--full-info', type=Path, required=True)
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--videos', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        for group, count, dimensions in build_plan(
                args.prompt_file, args.full_info, args.dimensions, suite=args.suite):
            print(f'{group}\t{count}\t{",".join(dimensions)}')
    else:
        if args.dimensions is not None:
            parser.error('--dimensions is only supported by plan')
        if not args.group or not args.videos:
            parser.error('Video validation requires --group and --videos')
        missing = missing_videos(args.prompt_file, args.group, args.videos, args.full_info)
        if args.command == 'missing-videos':
            print(len(missing))
        elif missing:
            raise ValueError(f'Missing or invalid videos for {args.group}: {missing[:5]}')
        else:
            print(f'VIDEOS_OK:{args.group}:frames={PIXEL_FRAMES}:fps={FPS}')


if __name__ == '__main__':
    main()

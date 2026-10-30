"""Merge VBench dimension results and report the official score."""
import argparse
import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.vbench.protocol import (SAMPLES_PER_PROMPT, LATENT_FRAMES, PIXEL_FRAMES,
                                        FPS, SUITES, expected_videos, resolve_dimensions,
                                        selected_groups, selection_id)

# Vchitect/VBench scripts/constant.py: normalize first, then apply weights.
BOUNDS = {
    'subject_consistency': (0.1462, 1.0), 'background_consistency': (0.2615, 1.0),
    'temporal_flickering': (0.6293, 1.0), 'motion_smoothness': (0.7060, 0.9975),
    'dynamic_degree': (0.0, 1.0), 'aesthetic_quality': (0.0, 1.0),
    'imaging_quality': (0.0, 1.0), 'object_class': (0.0, 1.0),
    'multiple_objects': (0.0, 1.0), 'human_action': (0.0, 1.0),
    'color': (0.0, 1.0), 'spatial_relationship': (0.0, 1.0),
    'scene': (0.0, 0.8222), 'appearance_style': (0.0009, 0.2855),
    'temporal_style': (0.0, 0.364), 'overall_consistency': (0.0, 0.364),
}


def summarize(results, dimensions=None, *, suite='quality'):
    selected = resolve_dimensions(dimensions, suite=suite)
    expected = set(selected)
    if set(results) != expected:
        raise ValueError(f'Expected {sorted(expected)}, got {sorted(results)}')
    raw = {key: float(results[key][0] if isinstance(results[key], list) else results[key])
           for key in selected}
    if not all(math.isfinite(value) for value in raw.values()):
        raise ValueError('VBench scores must be finite')
    normalized = {key: (value - BOUNDS[key][0]) / (BOUNDS[key][1] - BOUNDS[key][0])
                  for key, value in raw.items()}
    weights = {key: 0.5 if key == 'dynamic_degree' else 1.0 for key in raw}
    score = sum(normalized[key] * weights[key] for key in raw) / sum(weights.values())
    result = {'raw': raw, 'normalized': normalized, 'weights': weights,
              'protocol': {'samples_per_prompt': SAMPLES_PER_PROMPT,
                           'num_output_frames_latent': LATENT_FRAMES,
                           'pixel_frames': PIXEL_FRAMES, 'fps': FPS}}
    if selection_id(selected, suite=suite) is None:
        result[f'{suite}_score'] = score
    else:
        result['dimensions'] = list(selected)
        result['subset_score'] = score
    return result


def merge_results(inputs, expected):
    merged = {}
    for input_name in inputs:
        path = Path(input_name)
        data = json.loads(path.read_text(encoding="utf-8"))
        overlap = set(merged) & set(data)
        if overlap:
            raise ValueError(f"Duplicate dimensions {sorted(overlap)} in {path}")
        merged.update(data)

    if set(merged) != expected:
        raise ValueError(
            f"Expected dimensions {sorted(expected)}, got {sorted(merged)}"
        )
    return merged


def validate_result(path, dimension, prompt_file, group, videos, full_info):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(data) != {dimension}:
        raise ValueError(f'Expected only {dimension}: {path}')
    value = data[dimension]
    if not isinstance(value, list) or len(value) != 2 or not math.isfinite(float(value[0])):
        raise ValueError(f'Invalid score: {path}')
    records = value[1]
    if not isinstance(records, list):
        raise ValueError(f'Invalid per-video score records: {path}')
    paths = [Path(row['video_path']).resolve() for row in records]
    expected = {p.resolve() for p in expected_videos(prompt_file, group, videos, full_info)}
    if len(paths) != len(expected) or set(paths) != expected:
        raise ValueError(f'Incomplete/duplicate video scores: {path}')
    if any(not p.is_file() or p.stat().st_mtime_ns > Path(path).stat().st_mtime_ns for p in expected):
        raise ValueError(f'Videos changed after scoring: {path}')
    return data


def find_result(directory, dimension, prompt_file, group, videos, full_info):
    for path in sorted(Path(directory).glob('*_eval_results.json'), reverse=True):
        try:
            validate_result(path, dimension, prompt_file, group, videos, full_info)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('find-result', 'summarize'))
    parser.add_argument('--suite', choices=SUITES, default='quality')
    parser.add_argument('--dimensions', help='Optional comma-separated metric subset within --suite.')
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--full-info', type=Path, required=True)
    parser.add_argument('--videos', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dimension')
    parser.add_argument('--group')
    parser.add_argument('inputs', nargs='*')
    args = parser.parse_args()
    if args.command == 'find-result':
        if args.dimensions is not None:
            parser.error('--dimensions is only supported by summarize')
        if not args.dimension or not args.group:
            parser.error('find-result requires --dimension and --group')
        result = find_result(args.output_dir, args.dimension, args.prompt_file,
                             args.group, args.videos, args.full_info)
        if result:
            print(result)
        return
    expected = set(resolve_dimensions(args.dimensions, suite=args.suite))
    merged = merge_results(args.inputs, expected)
    for group, dimensions in selected_groups(args.dimensions, suite=args.suite).items():
        for dimension in dimensions:
            matching = [p for p in args.inputs if dimension in json.loads(Path(p).read_text(encoding='utf-8'))]
            validate_result(matching[0], dimension, args.prompt_file,
                            group, args.videos, args.full_info)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from evaluation.generator import write_json
    write_json(args.output_dir / 'combined_eval_results.json', merged)
    result = summarize(merged, args.dimensions, suite=args.suite)
    write_json(args.output_dir / 'summary.json', result)
    score_key = 'subset_score' if 'subset_score' in result else args.suite + '_score'
    label = args.suite.upper() + ('_SUBSET' if score_key == 'subset_score' else '')
    print(f'{label}_SCORE:{result[score_key]:.8f}')


if __name__ == '__main__':
    main()

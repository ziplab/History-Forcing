"""Paired PlayWorld OOS subset report using the upstream score and gate rules."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))
from evaluation.io import write_text  # noqa: E402
from evaluation.playworld.adapter import _write  # noqa: E402
from evaluation.playworld.upstream.metrics.aggregate import (  # noqa: E402
    OE_OUT_OF_SIGHT_TASKS, _finite, aggregate_fail_one_oe_split, embedded_review,
    normalized_score, score_key,
)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _phase(root, ids, *, model, fps, max_video_seconds, agent_model, protocol):
    prepared = protocol in {"training_camera_trajectory_v1", "preset_camera_trajectory_v1"}
    score_root = root.with_name(root.name + "_score")
    selected_ids = set(ids)
    # Older records may contain any JSON value returned by the judge. Keep the
    # registered sample, but pass only objects to the upstream aggregation.
    scores = {key: value if isinstance(value, dict) else {}
              for key, value in _read(score_root / "scores.json").items()}
    completion_path = score_root / "gc_task_completion.json"
    completion = _read(completion_path) if completion_path.is_file() else {}
    reviews = completion.get("results", completion)
    manifest = [item for item in _read(score_root / "videos_manifest.json")
                if item['task'] in selected_ids]
    judge = _read(score_root / "judge.json")
    if len({item["task"] for item in manifest}) != len(manifest):
        raise ValueError(f"Duplicate scored OOS tasks in {root}")
    by_task = {item["task"]: item for item in manifest
               if not item.get("isPlaceholder") and not item.get("isFullProcess")}
    output = {}
    for task in ids:
        item = by_task.get(task)
        if item is not None and item["group"] != "OE":
            raise ValueError(f"Expected an OE task: {task}")
        result = (scores.get(score_key(item)) or {}) if item is not None else {}
        raw = normalized_score(score_key(item), result) if item is not None else None
        gate = embedded_review(result, "OE", str(item.get("view"))) if item is not None else None
        if gate is None and item is not None:
            review = reviews.get(score_key(item))
            if (isinstance(review, dict) and review.get("verdict") in {"pass", "partial", "fail"}
                    and not review.get("error")):
                gate = review
        passed = _finite(gate.get("instruction_following_score")) == 1 if gate is not None else None
        status = ("missing_sample" if item is None else "invalid_score" if raw is None
                  else "missing_gate" if gate is None else "valid")
        categories = result.get("category_scores_1_to_5") or result.get("categories") or {}
        valid_categories = {}
        if raw is not None and isinstance(categories, dict):
            for name, value in categories.items():
                if not isinstance(value, dict):
                    continue
                score, weight = _finite(value.get("score")), _finite(value.get("weight"))
                if score is not None and (weight is None or weight > 0):
                    valid_categories[name] = score
        dynamicity = (result.get("oe_dynamicity_check") or {}) if raw is not None else {}
        if not isinstance(dynamicity, dict):
            dynamicity = {}
        record = {
            "raw": raw, "gate_pass": passed,
            "instruction_verdict": gate["verdict"] if gate is not None else None,
            "official_effective": (1.0 if item is None else
                                   (raw if passed else 1.0) if status == "valid" else None),
            "score_status": status, "seconds": None,
            "category_scores": valid_categories,
            "evolution_quality_level": _finite(dynamicity.get("evolution_quality_level_1_to_5")),
            "phase_comparison_applicable": dynamicity.get("phase_comparison_applicable"),
            "player_protocol": None, "agent_thinking": None,
            "trajectory_inputs": None, "inference_options": None,
            "sampling_manifest": None, "seed": None,
            "generation_metadata_available": False,
        }
        output[task] = record
        # Generation settings must match even when the judge response is invalid.
        if item is None:
            continue
        video = root / task / f"{task}.mp4"
        if not (video.parent / "result.json").is_file():
            continue
        episode = _read(video.parent / "result.json")
        if (episode.get("status") != "completed" or episode.get("model") != model or
                episode.get("fps") != fps or
                (not prepared and episode.get("max_video_seconds") != max_video_seconds) or
                Path(episode.get("video_path", "")).resolve() != video.resolve()):
            raise ValueError(f"PlayWorld video protocol mismatch: {task}")
        if prepared:
            if (episode.get("player_protocol") != protocol or
                    episode.get("trajectory_inputs", {}).get("task_id") != task or
                    not (episode.get("sampling_manifest") or episode.get("inference_options"))):
                raise ValueError(f"Preset camera trajectory protocol mismatch: {task}")
        elif (episode.get("agent_provider") != "seed-lite" or
              episode.get("agent_model") != agent_model or
              episode.get("agent_source") != "upstream/Agent_player/SANA-WM/run_agent.py"):
            raise ValueError(f"PlayWorld closed-loop protocol mismatch: {task}")
        episode_seed = episode.get("seed")
        trajectory_seed = (episode.get("trajectory_inputs") or {}).get("seed")
        if (episode_seed is not None and trajectory_seed is not None
                and episode_seed != trajectory_seed):
            raise ValueError(f"Episode and trajectory seeds differ: {task}")
        seed = trajectory_seed if episode_seed is None else episode_seed
        seconds = _duration(video)
        if not prepared and max_video_seconds is not None and seconds > max_video_seconds + .001:
            raise ValueError(f"Video exceeds {max_video_seconds:g} seconds: {task}={seconds:.3f}s")
        record.update({
            "seconds": round(seconds, 3),
            "player_protocol": episode.get("player_protocol"),
            "agent_thinking": episode.get("agent_thinking", "disabled"),
            "trajectory_inputs": episode.get("trajectory_inputs"),
            "inference_options": episode.get("inference_options"),
            "sampling_manifest": episode.get("sampling_manifest"),
            "seed": seed,
            "generation_metadata_available": True,
        })
    return output, judge, manifest, scores, completion


def compare(selection, baseline, after, *, model="backbones.reworld:ReWorld",
            fps=24, max_video_seconds=20,
            agent_model="doubao-seed-2-0-mini-260428", protocol="closed_loop"):
    if selection.suffix == ".parquet":
        from evaluation.playworld.adapter import preset_metadata

        selected = preset_metadata(selection)["selection"]
    else:
        selected = _read(selection)
    ids = selected["task_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected distinct selected OOS tasks")
    if not set(ids) <= OE_OUT_OF_SIGHT_TASKS:
        raise ValueError("Selection contains tasks outside the official OOS split")
    options = dict(model=model, fps=fps, max_video_seconds=max_video_seconds,
                   agent_model=agent_model, protocol=protocol)
    before, before_judge, before_manifest, before_scores, before_completion = _phase(baseline, ids, **options)
    trained, after_judge, after_manifest, after_scores, after_completion = _phase(after, ids, **options)
    if before_judge != after_judge:
        raise ValueError("Before/after judge settings differ")
    comparable = [task for task in ids if before[task]["generation_metadata_available"]
                  and trained[task]["generation_metadata_available"]]
    if any(before[task]["player_protocol"] != trained[task]["player_protocol"]
           for task in comparable):
        raise ValueError("Before/after Player protocols differ")
    if any(before[task]["agent_thinking"] != trained[task]["agent_thinking"] for task in comparable):
        raise ValueError("Before/after Agent thinking settings differ")
    if any(before[task]["trajectory_inputs"] != trained[task]["trajectory_inputs"] or
           before[task]["seed"] != trained[task]["seed"] for task in comparable):
        raise ValueError("Before/after camera, prompt, seed or sampling settings differ")
    unverified_sampling = []
    for task in ids:
        if task not in comparable:
            unverified_sampling.append(task)
            continue
        left, right = before[task]["sampling_manifest"], trained[task]["sampling_manifest"]
        observed = all(isinstance(record, dict) and record.get("schema_version") == 1
                       and record.get("status") == "observed"
                       and record.get("seed") is not None
                       and record.get("effective", {}).get("schedules") for record in (left, right))
        if observed:
            if left["seed"] != right["seed"]:
                raise ValueError(f"Before/after actual sampling seeds differ: {task}")
            if any(phase["seed"] is not None and record["seed"] != phase["seed"]
                   for record, phase in ((left, before[task]), (right, trained[task]))):
                raise ValueError(f"Actual sampling seed differs from episode inputs: {task}")
            if left["effective"] != right["effective"]:
                raise ValueError(f"Before/after effective sampling settings differ: {task}")
        else:
            unverified_sampling.append(task)
            # Old YAML-derived options still expose known mismatches, but their
            # equality cannot prove which solver/grid was used by either model.
            if not left and not right and before[task]["inference_options"] != trained[task]["inference_options"]:
                raise ValueError("Before/after camera, prompt, seed or sampling settings differ")
    rows = [{
        "task_id": task, "baseline": before[task], "after": trained[task],
        "delta": round(trained[task]["official_effective"] -
                       before[task]["official_effective"], 4)
                 if task in comparable and before[task]["official_effective"] is not None
                 and trained[task]["official_effective"] is not None else None,
    } for task in ids]
    def paired_summary(rows):
        deltas = [row["delta"] for row in rows if row["delta"] is not None]
        return {"paired_delta_mean": sum(deltas) / len(deltas) if deltas else None,
                "paired_count": len(deltas)}
    def summary(phase, manifest, scores, completion):
        values = list(phase.values())
        items = [item for item in manifest if item["task"] in phase
                 and not item.get("isPlaceholder") and not item.get("isFullProcess")]
        models = {item["model"] for item in items}
        if len(models) > 1:
            raise ValueError("Scored OOS tasks contain multiple models")
        label = next(iter(models), "unscored")
        official = aggregate_fail_one_oe_split(
            items, scores, completion, models=((label, label),),
        )[0]["out_of_sight"]
        # The official aggregate pads all 43 OOS tasks. Remove only the padding
        # outside this selected subset; retain its handling of selected failures.
        padding = len(OE_OUT_OF_SIGHT_TASKS - phase.keys())
        denominator = official["denominator"] - padding
        mean = official["average"]
        if padding:
            mean = (mean * official["denominator"] - padding) / denominator if denominator else None
        raw = [v["raw"] for v in values if v["raw"] is not None]
        durations = [v["seconds"] for v in values if v["seconds"] is not None]
        evolution = [v["evolution_quality_level"] for v in values
                     if v["evolution_quality_level"] is not None]
        categories = sorted({name for value in values
                             for name in value["category_scores"]})
        return {
            "official_oos_mean": mean,
            "raw_mean": sum(raw) / len(raw) if raw else None,
            "raw_count": len(raw),
            "gate_pass": official["non_fail"],
            "instruction_verdict_counts": {
                verdict: sum(v["score_status"] == "valid" and v["instruction_verdict"] == verdict
                             for v in values)
                for verdict in ("pass", "partial", "fail")
            },
            "count": denominator, "selected_count": len(values),
            "score_status_counts": {status: sum(v["score_status"] == status for v in values)
                                    for status in ("valid", "missing_sample", "invalid_score", "missing_gate")},
            "max_video_seconds": max(durations) if durations else None,
            "evolution_quality_mean": sum(evolution) / len(evolution) if evolution else None,
            "evolution_quality_count": len(evolution),
            "phase_comparison_applicable": sum(
                v["phase_comparison_applicable"] == "yes" for v in values),
            "raw_category_means": {
                name: {"mean": sum(scores) / len(scores), "count": len(scores)}
                for name in categories
                if (scores := [v["category_scores"][name] for v in values
                               if name in v["category_scores"]])
            },
        }
    before_summary = summary(before, before_manifest, before_scores, before_completion)
    after_summary = summary(trained, after_manifest, after_scores, after_completion)
    category_names = before_summary["raw_category_means"].keys() & after_summary["raw_category_means"].keys()
    paired_categories = {}
    for name in sorted(category_names):
        differences = [trained[task]["category_scores"][name] -
                       before[task]["category_scores"][name]
                       for task in ids if name in before[task]["category_scores"]
                       and name in trained[task]["category_scores"]]
        if differences:
            paired_categories[name] = {
                "mean_delta": sum(differences) / len(differences),
                "count": len(differences),
            }
    subsets = {}
    if 'training_task_ids' in selected:
        training_ids = set(selected['training_task_ids'])
        if not training_ids <= set(ids):
            raise ValueError('Training tasks are outside the scored selection')
        for name, members in [('training_images', training_ids),
                              ('held_out_images', set(ids) - training_ids)]:
            if members:
                subsets[name] = dict(
                    baseline=summary({task: before[task] for task in members}, before_manifest, before_scores,
                                     before_completion),
                    after=summary({task: trained[task] for task in members}, after_manifest, after_scores,
                                  after_completion),
                    **paired_summary([row for row in rows if row["task_id"] in members]))
    return {
        "judge": before_judge, "protocol": protocol,
        "sampling_comparison": "unverified" if unverified_sampling else "verified",
        "unverified_sampling_tasks": unverified_sampling,
        "official_closed_loop": protocol == "closed_loop",
        "in_sample": selected.get("in_sample", True),
        "subsets": subsets,
        "baseline": before_summary, "after": after_summary,
        **paired_summary(rows),
        "paired_raw_category_deltas": paired_categories,
        "tasks": rows,
    }


def _summary_table(summaries):
    phases = list(summaries.values())
    lines = ["| 指标 | " + " | ".join(summaries) + " |",
             "|---|" + "---:|" * len(phases)]

    def row(label, values):
        cells = ["—" if value is None else f"{value:.3f}" if isinstance(value, float)
                 else str(value) for value in values]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    labels = {'official_oos_mean': 'OOS 得分（指令失败计 1 分）',
              'raw_mean': '门控前得分', 'raw_count': '原始评分数量',
              'gate_pass': '指令 pass/partial 数量', 'count': '计分分母',
              'selected_count': '选定样本数', 'evolution_quality_mean': '动态演化质量',
              'evolution_quality_count': '动态演化评分数量',
              'phase_comparison_applicable': '可比较回访阶段的样本数',
              'max_video_seconds': '最长视频秒数', 'paired_delta_mean': '相对 Base 配对差值',
              'paired_count': '配对样本数', 'sampling_comparison': '采样一致性'}
    keys = dict.fromkeys(key for phase in phases for key, value in phase.items()
                         if not isinstance(value, dict))
    for key in keys:
        row(labels.get(key, key), [phase.get(key) for phase in phases])
    for group in ('instruction_verdict_counts', 'score_status_counts'):
        names = sorted({name for phase in phases for name in phase[group]})
        for name in names:
            row(f'{group}/{name}', [phase[group].get(name, 0) for phase in phases])
    for name in sorted({name for phase in phases for name in phase['raw_category_means']}):
        values = [phase['raw_category_means'].get(name) for phase in phases]
        row(name, [f"{value['mean']:.3f} (n={value['count']})" if value else None
                   for value in values])
    return lines


def write_comparison_table(report_paths, output):
    """Combine saved paired reports without generating or rescoring videos."""
    if not report_paths or 'Base' in report_paths:
        raise ValueError('Expected checkpoint reports with distinct labels other than Base')
    reports = {label: _read(Path(path)) for label, path in report_paths.items()}
    first = next(iter(reports.values()))
    reference = [(row['task_id'], row['baseline']) for row in first['tasks']]
    for report in reports.values():
        if (any(report[key] != first[key] for key in
                ('judge', 'protocol', 'official_closed_loop', 'in_sample', 'baseline'))
                or [(row['task_id'], row['baseline']) for row in report['tasks']] != reference):
            raise ValueError('Checkpoint reports use different baselines, tasks or protocols')
    summaries = {'Base': first['baseline']}
    summaries.update({label: {**report['after'], **{key: report[key] for key in
        ('paired_delta_mean', 'paired_count', 'sampling_comparison')}}
        for label, report in reports.items()})
    result = dict(judge=first['judge'], protocol=first['protocol'],
                  official_closed_loop=first['official_closed_loop'], in_sample=first['in_sample'],
                  models=summaries, reports={label: str(path) for label, path in report_paths.items()})
    lines = ['# PlayWorld OOS 权重比较', '',
             f"Judge: `{first['judge']['model']}` · {first['baseline']['selected_count']} 条样本", '',
             f"Protocol: `{first['protocol']}`", '', *_summary_table(summaries)]
    output = Path(output)
    _write(output, result)
    write_text(output.with_suffix('.md'), '\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="backbones.reworld:ReWorld")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-video-seconds", type=float, default=20)
    parser.add_argument("--agent-model", default="doubao-seed-2-0-mini-260428")
    parser.add_argument("--protocol", choices=("closed_loop", "training_camera_trajectory_v1",
                                              "preset_camera_trajectory_v1"),
                        default="closed_loop")
    args = parser.parse_args()
    result = compare(args.selection, args.baseline, args.after, model=args.model,
                     fps=args.fps, max_video_seconds=args.max_video_seconds,
                     agent_model=args.agent_model, protocol=args.protocol)
    _write(args.output, result)
    before, after = result['baseline'], result['after']
    if result['official_closed_loop']:
        protocol_note = "使用闭环 Agent 轨迹。"
    elif result['in_sample']:
        protocol_note = "使用训练图像与预设相机轨迹；这是训练样本上的比较，不是官方闭环 Agent 得分。"
    else:
        protocol_note = "使用独立评测图像与预设相机轨迹；这是离线评测，不是官方闭环 Agent 得分。"
    sampling_note = (f"采样一致性未验证：{len(result['unverified_sampling_tasks'])} 条样本缺少有效的实际采样记录。"
                     if result['sampling_comparison'] == 'unverified' else
                     "已核对 Base 与训练后的实际采样设置一致。")
    lines = ["# PlayWorld OOS 配对比较", "",
             f"Judge: `{result['judge']['model']}` · {before['selected_count']} 条样本", "",
             protocol_note, sampling_note, "", *_summary_table({'Base': before, '训练后': after})]
    lines += ["", "| 样本 | Base | 训练后 | 差值 | 指令判断 Base → 训练后 |",
              "|---|---:|---:|---:|---|"]
    for row in result['tasks']:
        effective = [phase['official_effective'] for phase in (row['baseline'], row['after'])]
        scores = [f"{value:.3f}" if value is not None else "—" for value in effective]
        delta = f"{row['delta']:+.3f}" if row['delta'] is not None else "—"
        lines.append(f"| {row['task_id']} | {scores[0]} | {scores[1]} | {delta} | "
                     f"{row['baseline']['instruction_verdict']} → {row['after']['instruction_verdict']} |")
    write_text(args.output.with_suffix('.md'), '\n'.join(lines) + '\n')
    print(json.dumps({key: result[key] for key in
                      ("baseline", "after", "paired_delta_mean", "sampling_comparison",
                       "unverified_sampling_tasks")}, indent=2))


if __name__ == "__main__":
    main()

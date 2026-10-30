#!/usr/bin/env python3
"""Shared YAML configuration and launch helpers for the project pipelines."""

from config import PROJECT_ROOT, REPO_ROOT, _path_from_repo, _run_config, _select, add_sample_arguments, load_config, sample_count, stage_argument, stage_values, training_recipe
from utils import find_generated_files

import argparse
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from omegaconf.errors import OmegaConfBaseException

from recipes import RECIPE_NAMES


def sanitize_omp_num_threads(environment=None):
    """Replace an invalid OpenMP thread count before native libraries load."""
    environment = os.environ if environment is None else environment
    try:
        valid = int(environment.get("OMP_NUM_THREADS", "1")) > 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        environment["OMP_NUM_THREADS"] = "1"
    return environment


def launch_environment(config):
    """Resolve worker resources once; explicit environment values override YAML."""
    env = os.environ.copy()
    threads = _select(config, 'environment.OMP_NUM_THREADS')
    if threads is not None:
        env.setdefault('OMP_NUM_THREADS', str(threads))
    sanitize_omp_num_threads(env)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (
        str(PROJECT_ROOT / 'src'), str(PROJECT_ROOT), env.get('PYTHONPATH'),
    )))
    env.pop('PREPARE_EVALUATION', None)
    cache_dir = _select(config, 'environment.vbench_cache_dir')
    if cache_dir:
        env['VBENCH_CACHE_DIR'] = str(_path_from_repo(cache_dir))
    for name, default in (('HF_HOME', 'models/.cache/huggingface'),
                          ('TORCH_HOME', 'models/.cache/torch')):
        directory = _select(config, f'environment.{name}', default)
        env.setdefault(name, str(_path_from_repo(directory)))
    try:
        processes = int(env.get('NPROC_PER_NODE', _select(config, 'launcher.NPROC_PER_NODE', 1)))
    except (TypeError, ValueError) as error:
        raise ValueError('NPROC_PER_NODE must be an integer') from error
    if processes <= 0:
        raise ValueError('NPROC_PER_NODE must be positive')
    env['NPROC_PER_NODE'] = str(processes)
    return env


def _signal_group(process, sig):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _process_start(pid):
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except (FileNotFoundError, PermissionError):
        return None
    end = stat.rfind(")")
    fields = stat[end + 2:].split()
    return fields[19] if end >= 0 and len(fields) > 19 else None


def _descendants(pid):
    """Read the Linux process tree before torchrun workers become orphaned."""
    descendants = {}
    pending = [pid]
    seen = {pid}
    while pending:
        parent = pending.pop()
        children_path = (
            Path("/proc") / str(parent) / "task" / str(parent) / "children"
        )
        try:
            children = [
                int(value) for value in children_path.read_text().split()
            ]
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        for child in children:
            if child not in seen:
                seen.add(child)
                start = _process_start(child)
                if start is not None:
                    descendants[child] = start
                pending.append(child)
    return descendants


def _alive(processes):
    alive = {}
    for pid, start in processes.items():
        if _process_start(pid) == start:
            alive[pid] = start
    return alive


def _signal_pids(processes, sig):
    for pid, start in reversed(list(processes.items())):
        if _process_start(pid) != start:
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _interrupt(process):
    """Stop torchrun through its agent, then bound forced cleanup time."""
    if process.poll() is not None:
        return
    descendants = _descendants(process.pid) if os.name == "posix" else {}
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        remaining = _alive(descendants)
        if remaining:
            _signal_pids(remaining, signal.SIGTERM)
            time.sleep(1)
            _signal_pids(_alive(remaining), signal.SIGKILL)
        return
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass

    descendants.update(_descendants(process.pid))
    _signal_pids(descendants, signal.SIGTERM)
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    _signal_pids(_alive(descendants), kill_signal)
    _signal_group(process, kill_signal)
    try:
        process.wait(timeout=1)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass


def _run(script, config_path, extra_args, env, processes=None, *, log_path=None):
    command = [sys.executable]
    if processes is not None and not any(arg in {'-h', '--help'} for arg in extra_args):
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={processes}",
        ]
    command += ([str(Path(__file__).parent / script)] if str(script).endswith('.py')
                else ['-m', str(script)])
    if config_path is not None:
        command += ["--config", str(config_path)]
    command += extra_args
    with log_path.open('wb') if log_path is not None else nullcontext(None) as log:
        process = subprocess.Popen(
            command, env=env, cwd=REPO_ROOT,
            stdout=log, stderr=subprocess.STDOUT if log is not None else None,
            start_new_session=(os.name == "posix"),
        )
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            print("\nStopping processes ...", flush=True)
            _interrupt(process)
            raise SystemExit(130) from None
    if returncode:
        raise SystemExit(returncode)


def _run_evaluation(config_path, extra_args, env, processes, *, evaluator=None, log_path=None):
    if evaluator:
        # Task evaluators coordinate preparation, generation, scoring and
        # aggregation once; each task owns its worker launches.
        _run("evaluate.py", config_path, extra_args, env, log_path=log_path)
        return
    if not any(arg in {"-h", "--help"} for arg in extra_args):
        prepare_env = env.copy()
        prepare_env["PREPARE_EVALUATION"] = "1"
        _run("evaluate.py", config_path, extra_args, prepare_env)
    _run("evaluate.py", config_path, extra_args, env, processes)


def pack_data(config, extra_args, env, *, recipe=None, section='train', manifest=None,
              inference_args=None):
    """Pack for the selected stage, matching inference's native CLI overrides."""
    values = {
        'model': stage_argument(config, section, 'model', recipe=recipe),
        'model-dir': stage_argument(config, section, 'model_dir', recipe=recipe),
        'manifest': (str(_path_from_repo(manifest)) if manifest is not None else
                     _select(config, 'paths.train_manifest' if section == 'train' else 'inference.parquet')),
        'output-dir': stage_argument(config, section, 'data_path', recipe=recipe),
    }
    if inference_args is not None:
        parser = argparse.ArgumentParser(add_help=False)
        for flag, key in (('model', 'model'), ('model-dir', 'model-dir'), ('data-path', 'output-dir')):
            parser.add_argument(f'--{flag}', dest=key, default=values[key])
        selected, _ = parser.parse_known_args(inference_args)
        values.update(vars(selected))
    flags = [item for name, value in values.items() if value is not None
             for item in (f'--{name}', str(value))]
    _run('../tools/pack_video.py', None, [*flags, *extra_args], env)


def _experiment(config_path, config, env, processes, recipe):
    """Evaluate configured benchmarks at baseline and saved checkpoints."""
    from engine.evaluation import load_checkpoint_evaluator

    train = stage_values(config, 'train', recipe=recipe)
    output = _path_from_repo(train['output_dir'])
    evaluator = load_checkpoint_evaluator(train.get('evaluator'), train.get('eval_config') or config_path)
    try:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'train.exit').unlink(missing_ok=True)
        start_step = 0
        mode = stage_argument(config, 'train', 'checkpoint_mode', 'resume', recipe=recipe)
        if evaluator is not None and mode != 'weights':
            from backbones import DEFAULT_MODEL, load_backend
            from engine.runtime import read_training_state

            backend = load_backend(stage_argument(config, 'train', 'model', DEFAULT_MODEL, recipe=recipe))
            checkpoint = backend.checkpoint_file(_path_from_repo(train['checkpoint']))
            state = read_training_state(checkpoint)
            start_step = 0 if state is None else state['global_step']
            del state
        pack_data(config, ['--replace-existing'], env, recipe=recipe)
        if evaluator is not None:
            print(f"Baseline evaluation log: {output / 'baseline.log'}", flush=True)
            _run_evaluation(config_path, [], env, processes,
                            evaluator=evaluator, log_path=output / 'baseline.log')
        print(f"Training log: {output / 'train.log'}", flush=True)
        status = 0
        try:
            _run('train.py', config_path, ['--recipe', recipe], env, processes,
                 log_path=output / 'train.log')
        except BaseException as error:
            status = error.code if isinstance(error, SystemExit) else 1
            raise
        finally:
            (output / 'train.exit').write_text(f'{status}\n', encoding='utf-8')
        if evaluator is not None:
            maximum, interval = int(train['max_train_steps']), int(train['save_every'])
            steps = list(range(interval, maximum, interval)) + [maximum]
            evaluator.finalize(output, steps, start_step=start_step)
    finally:
        if evaluator is not None:
            evaluator.close()


def _evaluation_overrides(config, extra_args):
    """Evaluate the selected generator's outputs, retaining an explicit GT source."""
    from backbones import DEFAULT_MODEL

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=stage_argument(config, "inference", "model", DEFAULT_MODEL))
    for name in ("model_dir", "data_path"):
        value = stage_argument(config, "inference", name)
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path,
                            default=None if value is None else _path_from_repo(value))
    add_sample_arguments(parser)
    for name in ("num_samples", "sample_index", "samples_per_prompt"):
        value = stage_argument(config, "inference", name, parser.get_default(name))
        parser.set_defaults(**{name: sample_count(value) if name == "num_samples" else int(value)})
    value = stage_argument(config, "inference", "output_dir")
    parser.add_argument("--output-dir", type=Path,
                        default=None if value is None else _path_from_repo(value))
    args, _ = parser.parse_known_args(extra_args)

    forwarded = ["--model", args.model]
    for name in ("model_dir", "data_path"):
        if name == "data_path" and _select(config, "evaluation.data_path") is not None:
            continue
        value = getattr(args, name)
        if value is not None:
            forwarded += [f"--{name.replace('_', '-')}", str(value.resolve())]
    for name in ("num_samples", "sample_index", "samples_per_prompt"):
        value = getattr(args, name)
        forwarded += [f"--{name.replace('_', '-')}", "all" if value is None else str(value)]
    if args.output_dir is not None:
        forwarded += ["--latents-dir", str(args.output_dir.resolve())]
    return forwarded


def _evaluation_processes(config, extra_args, maximum):
    """Match ranks to native prompt shards or VBench video shards."""
    parser = argparse.ArgumentParser(add_help=False)
    add_sample_arguments(parser)
    for name in ('num_samples', 'sample_index', 'samples_per_prompt'):
        parser.set_defaults(**{name: _select(config, f'evaluation.{name}', parser.get_default(name))})
    parser.add_argument('--vbench', action=argparse.BooleanOptionalAction,
                        default=_select(config, 'evaluation.vbench', False))
    directory = _select(config, "evaluation.latents_dir")
    parser.add_argument(
        "--latents-dir",
        type=Path,
        default=None if directory is None else _path_from_repo(directory),
    )
    values, _ = parser.parse_known_args(extra_args)
    samples_per_prompt = values.samples_per_prompt
    count = values.num_samples
    processes = maximum
    if count is not None and count > 0 and samples_per_prompt > 0:
        processes = min(processes, count * (samples_per_prompt if values.vbench else 1))

    if values.latents_dir is not None:
        generated, mp4_files = find_generated_files(values.latents_dir)
        start = values.sample_index
        end = None if count is None else start + count
        sources = mp4_files if values.vbench else generated
        keys = {(prompt_index, sample_index) for prompt_index, sample_index in sources
                if prompt_index >= start and (end is None or prompt_index < end)
                and sample_index < samples_per_prompt}
        work = len(keys) if values.vbench else len({prompt_index for prompt_index, _ in keys})
        processes = min(processes, max(1, work))
    return processes


def _sample_processes(config, section, extra_args, maximum):
    """Avoid launching empty inference ranks for a requested sample range."""
    parser = argparse.ArgumentParser(add_help=False)
    add_sample_arguments(parser)
    parser.set_defaults(num_samples=_select(config, f'{section}.num_samples'))
    values, _ = parser.parse_known_args(extra_args)
    if values.num_samples is None or values.num_samples <= 0:
        return maximum
    return min(maximum, values.num_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--pipeline-help", action="help", help="show launcher help and exit"
    )
    parser.add_argument(
        "stage",
        choices=("pack", "train", "inference", "evaluate", "experiment"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recipe", choices=RECIPE_NAMES)
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a YAML field; applies to all stages of an experiment')
    parser.add_argument('--parquet', type=Path, help='Pack this Parquet before inference')
    args, extra_args = parser.parse_known_args()
    if args.stage not in {"pack", "train"} and args.recipe is not None:
        parser.error("--recipe is only valid for pack and train; experiment uses train.recipe from its config")
    if args.parquet is not None and args.stage != 'inference':
        parser.error('--parquet is only valid for inference')
    if args.stage == 'experiment' and extra_args:
        parser.error('experiment accepts --set overrides; use individual stages for native CLI flags')
    try:
        config_path, config = load_config(args.config, args.overrides, recipe=args.recipe)
        evaluator = _select(config, 'evaluation.evaluator') or stage_argument(config, 'train', 'evaluator')
        env = launch_environment(config)
    except (
        FileNotFoundError, OmegaConfBaseException, TypeError, ValueError,
    ) as error:
        parser.error(str(error))

    nproc_per_node = int(env['NPROC_PER_NODE'])

    run_evaluation = _select(config, "pipeline.run_evaluation", 'evaluation' in config and not evaluator)
    if not isinstance(run_evaluation, bool):
        parser.error("pipeline.run_evaluation must be boolean")
    run_evaluation &= not any(arg in {"-h", "--help"} for arg in extra_args)

    if args.overrides or args.recipe is not None:
        config_path = _run_config(config_path, config, args.stage)

    if args.stage == 'pack':
        pack_data(config, extra_args, env)
    elif args.stage == 'experiment':
        _experiment(config_path, config, env, nproc_per_node, training_recipe(config))
    elif args.stage == "train":
        recipe = training_recipe(config)
        _run(
            "train.py", config_path,
            ["--recipe", recipe, *extra_args], env, nproc_per_node,
        )
    elif args.stage == "inference":
        if run_evaluation and evaluator:
            parser.error('Task evaluation requires the evaluate stage to register benchmark trajectories')
        if args.parquet is not None:
            extra_args += ['--parquet', str(_path_from_repo(args.parquet))]
        if (args.parquet is not None or _select(config, 'inference.parquet')) and not any(arg in {'-h', '--help'} for arg in extra_args):
            pack_data(config, ['--replace-existing'], env, section='inference',
                      manifest=args.parquet, inference_args=extra_args)
        _run(
            "inference.py",
            config_path,
            extra_args,
            env,
            _sample_processes(
                config, "inference", extra_args, nproc_per_node
            ),
        )
        if run_evaluation:
            evaluation_args = _evaluation_overrides(config, extra_args)
            _run_evaluation(
                config_path, evaluation_args, env,
                _evaluation_processes(config, evaluation_args, nproc_per_node),
            )
    elif args.stage == "evaluate":
        _run_evaluation(
            config_path, extra_args, env,
            _evaluation_processes(config, extra_args, nproc_per_node),
            evaluator=evaluator,
        )


if __name__ == "__main__":
    main()

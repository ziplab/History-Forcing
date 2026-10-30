"""Distributed optimizer/checkpoint engine for OPSD distillation."""

import time
from contextlib import closing
from dataclasses import asdict
from functools import partial

import torch
import torch.distributed as dist
from tqdm import tqdm

from backbones import load_backend
from scheduler import sampling_schedule_manifest
from .evaluation import build_checkpoint_evaluator
from .runtime import (
    TRAINER_STATE_NAME,
    ExperimentTracker,
    build_dataloader,
    build_checkpoint_identity,
    build_lr_scheduler,
    clip_grad_norm,
    close_dataloader,
    common_training_identity,
    configure_runtime,
    distributed_call,
    distributed_phase,
    distributed_strategy,
    export_training_model,
    initialize_training_base,
    load_training_state,
    maybe_no_sync,
    next_training_position,
    optimizer_parameter_groups,
    parameter_counts,
    precompute_text_embeddings,
    require_finite,
    restore_training_state,
    save_trainer_state,
    start_training_pass,
    step_optimizer,
    tracker_config,
    training_autocast,
    training_session,
    validate_model_assets,
    validate_training_state,
    wrap_training_model,
    write_training_checkpoint,
)


def _save_checkpoint(
    model,
    teacher,
    optimizer,
    scheduler,
    args,
    recipe,
    backend,
    step,
    data_pass,
    batch_index,
    checkpoint_identity,
    model_config,
    base_checkpoint,
    rank,
    world_size,
    device,
    checkpoint_group,
    dataset,
    evaluate=None,
):
    def write_payload(directory):
        export_training_model(
            directory, model, backend, model_config, base_checkpoint,
            checkpoint_identity, args.lora_rank, args.lora_alpha,
            rank, device, checkpoint_group,
            artifacts=lambda state: recipe.checkpoint_artifacts(
                backend.unwrap(model), state, teacher, args.lora_rank,
            ),
            merge=recipe.merge_student,
        )
        save_trainer_state(
            directory / TRAINER_STATE_NAME, model, optimizer, scheduler,
            rank, world_size, device, step, data_pass, batch_index,
            checkpoint_group,
        )
        return (tuple(recipe.adapter_files.values()) if args.lora_rank
                else (recipe.teacher_ema_file,))

    def evaluate_saved(directory, saved_step):
        # LMDB permits one environment per path in a process. The evaluator
        # may read the training pack; workers own separate process handles.
        distributed_call(dataset.close, checkpoint_group, "OPSD data release before evaluation")
        return evaluate(directory, saved_step)

    return write_training_checkpoint(
        args.output_dir, step, rank=rank, world_size=world_size,
        process_group=checkpoint_group, write_payload=write_payload,
        max_checkpoints=args.max_checkpoints,
        evaluate=evaluate_saved if evaluate is not None else None,
    )


def run_opsd_training(args, recipe):
    with training_session(args.seed) as session:
        return _run_opsd_training(args, recipe, *session)


def _run_opsd_training(args, recipe, rank, world_size, device, checkpoint_group, resources):

    def initialize_inputs():
        initialized_backend = load_backend(args.model)
        initialized_config = recipe.model_config(initialized_backend, args)
        validate_model_assets(
            args.model_dir, args.checkpoint, args.model, config=initialized_config,
        )
        dataset = resources.enter_context(closing(recipe.dataset(args.data_path, initialized_backend)))
        try:
            metadata = [dataset.metadata(index) for index in range(len(dataset))]
            prompts = [initialized_backend.opsd_prompt_group(recipe.prompts(row), args.guidance_scale) for row in metadata]
        finally:
            dataset.close()
        return (
            initialized_backend,
            initialized_config,
            dataset,
            metadata,
            prompts,
        )

    backend, model_config, dataset, metadata, prompt_groups = distributed_call(
        initialize_inputs,
        checkpoint_group,
        "OPSD input initialization",
    )
    group_ids = [(row['condition_frames'], row['target_frames'], row['revisit_start_latent']) for row in metadata]

    def initialize_loader():
        loader, sampler, info = build_dataloader(
            dataset, args.batch_size, args.gradient_accumulation,
            args.num_workers, args.seed, rank, world_size,
            collate_fn=list, group_ids=group_ids,
        )
        resources.callback(close_dataloader, loader)
        return loader, sampler, info

    dataloader, data_sampler, loader_info = distributed_call(
        initialize_loader,
        checkpoint_group, 'OPSD data loader initialization',
    )
    def initialize_sampler():
        schedule = recipe.schedule(device, backend, sampling_steps=args.sampling_steps,
                                   timestep_shift=args.timestep_shift, sampling_mode=args.sampling_mode)
        recipe.supervision_steps(
            schedule, args.num_supervised_denoising_steps,
            supervision_max_timestep=args.supervision_max_timestep,
        )
        return schedule, sampling_schedule_manifest(schedule)

    denoising, sampling_schedule = distributed_call(
        initialize_sampler,
        checkpoint_group,
        "OPSD sampler initialization",
    )
    sampler = type(denoising).__name__
    use_adapters = bool(args.lora_rank)
    configured_sampling_steps = (
        args.sampling_steps if type(args.sampling_steps) is int else list(args.sampling_steps)
    )
    distributed_call(
        lambda: args.output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0 else None,
        checkpoint_group,
        "OPSD output directory creation",
    )
    training_config = {
        "trainer": recipe.name,
        "identity_version": 4,
        "dataset": distributed_call(dataset.checkpoint_identity, checkpoint_group, "OPSD dataset identity"),
        'data_metadata': metadata,
        'samples': len(dataset),
        'rollout_groups': len(set(group_ids)),
        'micro_batches_per_pass': loader_info.micro_batches,
        **common_training_identity(args, backend),
        "sampler": sampler,
        "sampling_schedule": sampling_schedule,
        "sampling_steps": configured_sampling_steps,
        "timestep_shift": args.timestep_shift,
        "sampling_mode": sampling_schedule["sampling_mode"],
        "ema_decay": args.ema_decay,
        "mse_space": args.mse_space,
        "teacher_history": "fixed_camera",
        "history_guidance_scale": args.history_guidance_scale,
        "guidance_scale": args.guidance_scale,
        "text_guidance_formula": "negative + scale * (positive - negative)",
        "teacher_kv": asdict(args.teacher_kv),
        "sink_frames": args.sink_frames,
    }
    if args.guidance_scale != 1.0:
        training_config['negative_prompt'] = backend.negative_prompt
    if not use_adapters or args.sharding_strategy != 'NO_SHARD':
        training_config['sharding_strategy'] = args.sharding_strategy
    if args.num_supervised_denoising_steps is not None:
        training_config['num_supervised_denoising_steps'] = args.num_supervised_denoising_steps
    # Keep the legacy identity unchanged for checkpoints resumed with old configs.
    if args.supervision_max_timestep is not None:
        training_config['supervision_max_timestep'] = args.supervision_max_timestep
    checkpoint_identity = build_checkpoint_identity(
        args.model,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_target_modules,
        training_config,
        freeze_camera=args.freeze_camera,
    )
    resume = load_training_state(
        args.checkpoint, args.checkpoint_mode, rank, checkpoint_group
    )
    resume_config = None
    if resume is not None:
        resume_config = distributed_call(
            lambda: validate_training_state(
                resume, checkpoint_identity, model_config,
                backend.checkpoint_config(args.checkpoint), world_size,
                args.max_train_steps, args.checkpoint, loader_info,
                required_artifacts=(tuple(recipe.adapter_files.values()) if use_adapters
                                    else (recipe.teacher_ema_file,)),
            ),
            checkpoint_group,
            "OPSD checkpoint validation",
        )

    text_embeddings = precompute_text_embeddings(
        args.model_dir, prompt_groups, args.batch_size, args.output_dir, device,
        rank, world_size, list(iter(data_sampler)), model=args.model,
        error_group=checkpoint_group,
    )

    def initialize_model():
        torch.manual_seed(args.seed)
        initialized, base = initialize_training_base(
            backend, args.model_dir, args.checkpoint, model_config,
            lora_rank=args.lora_rank,
            saved_model=None if resume is None else resume_config["model"],
            output_dir=args.output_dir, rank=rank,
        )
        excluded = backend.freeze_camera_modules(initialized) if args.freeze_camera else ()
        recipe.apply_adapters(
            initialized,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            args.lora_target_modules,
            args.seed,
            exclude_modules=excluded,
        )
        if resume is not None and args.lora_rank:
            recipe.restore_artifacts(
                initialized, args.checkpoint
            )
        elif args.lora_checkpoint is not None:
            recipe.load_initial_adapter(
                initialized, backend, args.lora_checkpoint
            )
        backend.enable_gradient_checkpointing(initialized)
        initialized.eval()
        return initialized, base

    model, base_checkpoint = distributed_call(
        initialize_model,
        checkpoint_group,
        "OPSD model initialization",
    )
    if rank == 0 and resume is None and args.lora_checkpoint is not None:
        print(
            f"Loaded initial LoRA checkpoint: {args.lora_checkpoint}",
            flush=True,
        )
    trainable, total = parameter_counts(model)
    if rank == 0:
        print(
            f"trainable_parameters={trainable:,} total_parameters={total:,} "
            f"trainable_percent={100.0 * trainable / total:.4f} "
            f"rollout_batch_size={args.batch_size} "
            f"gradient_accumulation={args.gradient_accumulation} "
            f"world_size={world_size} effective_batch_size="
            f"{args.batch_size * args.gradient_accumulation * world_size} "
            f"dataset_samples={len(dataset)} rollout_groups={len(set(group_ids))} "
            f"repeated_samples_per_pass={loader_info.repeated_samples_per_pass}",
            flush=True,
        )
    model = wrap_training_model(
        model,
        device,
        parameter_efficient=use_adapters,
        transformer_layer_cls=backend.transformer_layer(),
        sharding_strategy=args.sharding_strategy,
    )
    strategy = distributed_strategy(model)
    if rank == 0:
        print(
            f"distributed_strategy={strategy}",
            flush=True,
        )

    def initialize_teacher():
        initialized = recipe.create_teacher(
            model,
            args.lora_rank,
            args.ema_decay,
        )
        if resume is not None:
            recipe.restore_teacher(initialized, args.checkpoint)
        return initialized

    teacher = distributed_call(
        initialize_teacher,
        checkpoint_group,
        "OPSD teacher initialization",
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)
    if resume is None:
        configure_runtime(args.seed, rank)
        global_step = data_pass = next_batch_index = 0
    else:
        global_step, data_pass, next_batch_index = restore_training_state(
            model,
            optimizer,
            scheduler,
            resume,
            rank,
            device,
            args.checkpoint.parent,
            checkpoint_group,
        )
    resume = None
    optimizer.zero_grad(set_to_none=True)
    tracker = distributed_call(
        lambda: (
            resources.enter_context(ExperimentTracker(
                args,
                tracker_config(
                    args,
                    training_mode=recipe.name,
                    dataset_samples=len(dataset),
                    rollout_groups=len(set(group_ids)),
                    repeated_samples_per_pass=loader_info.repeated_samples_per_pass,
                    sampler=sampler,
                    sampling_steps=configured_sampling_steps,
                    mse_space=args.mse_space,
                    num_supervised_denoising_steps=sum(recipe.supervision_steps(
                        denoising, args.num_supervised_denoising_steps,
                        supervision_max_timestep=args.supervision_max_timestep,
                    )),
                    supervision_max_timestep=args.supervision_max_timestep,
                    batch_size=args.batch_size,
                    gradient_accumulation=args.gradient_accumulation,
                    world_size=world_size,
                    effective_batch_size=(
                        args.batch_size
                        * args.gradient_accumulation
                        * world_size
                    ),
                ),
            ))
            if rank == 0 else None
        ),
        checkpoint_group,
        "OPSD tracker initialization",
    )
    evaluate = distributed_call(
        lambda: build_checkpoint_evaluator(
            args, resources=resources, model=model, backend=backend, device=device, rank=rank,
            world_size=world_size, process_group=checkpoint_group, tracker=tracker,
        ), checkpoint_group, "Checkpoint evaluator initialization",
    )
    save = partial(
        _save_checkpoint, model=model, teacher=teacher, optimizer=optimizer,
        scheduler=scheduler, args=args, recipe=recipe, backend=backend,
        checkpoint_identity=checkpoint_identity, model_config=model_config,
        base_checkpoint=base_checkpoint, rank=rank, world_size=world_size,
        device=device, checkpoint_group=checkpoint_group, dataset=dataset, evaluate=evaluate,
    )
    progress = (
        tqdm(
            total=args.max_train_steps,
            initial=global_step,
            desc=recipe.progress_description,
            dynamic_ncols=True,
        )
        if rank == 0 else None
    )
    if progress is not None:
        resources.callback(progress.close)
    last_saved = global_step
    start_time = time.monotonic()
    batch_iterator = None

    while global_step < args.max_train_steps:
        loss_sum = torch.zeros((), device=device)
        for accumulation_step in range(args.gradient_accumulation):
            if batch_iterator is None:
                batch_iterator = distributed_call(
                    lambda: start_training_pass(
                        dataloader, data_sampler, data_pass=data_pass,
                        batch_index=next_batch_index, batch_size=args.batch_size,
                        text_stores=(text_embeddings,),
                    ), checkpoint_group, 'OPSD data pass',
                )
            rows = distributed_call(lambda: next(batch_iterator), checkpoint_group, 'OPSD batch loading')

            def prepare_batch():
                prompt_batches = tuple(zip(*(backend.opsd_prompt_group(recipe.prompts(row), args.guidance_scale) for row in rows),
                                           strict=True))
                encoded = {prompts: text_embeddings.encode(prompts, next_batch_index)
                           for prompts in dict.fromkeys(prompt_batches)}
                contexts = tuple(encoded[prompts] for prompts in prompt_batches)
                return recipe.prepare_batch(rows, device, backend, model_config), contexts

            sample, contexts = distributed_call(prepare_batch, checkpoint_group, 'OPSD batch preparation')
            terms = recipe.loss_term_count(
                sample, denoising, args.num_supervised_denoising_steps,
                supervision_max_timestep=args.supervision_max_timestep,
            )
            if terms <= 0:
                raise ValueError('OPSD rollout produced no training terms')
            # Average each video's supervised steps first, then average the
            # accumulation window. Longer revisit segments get no extra weight.
            accumulated_terms = terms * args.gradient_accumulation
            rollout_index = (
                (
                    global_step * args.gradient_accumulation
                    + accumulation_step
                ) * world_size
                + rank
            )
            losses = iter(backend.opsd_loss_terms(recipe,
                model,
                teacher,
                sample,
                contexts,
                denoising,
                [args.seed + rollout_index * args.batch_size + i for i in range(len(rows))],
                mse_space=args.mse_space,
                use_adapters=use_adapters,
                sink_frames=args.sink_frames,
                num_supervised_denoising_steps=args.num_supervised_denoising_steps,
                supervision_max_timestep=args.supervision_max_timestep,
                history_guidance_scale=args.history_guidance_scale,
                guidance_scale=args.guidance_scale,
                teacher_kv=args.teacher_kv,
                validate=(
                    global_step == 0 and accumulation_step == 0
                ),
            ))
            try:
                with closing(losses):
                    for term in range(1, terms + 1):
                        synchronize = (
                            strategy == "fsdp"
                            or (
                                accumulation_step + 1
                                == args.gradient_accumulation
                                and term == terms
                            )
                        )
                        # DDP requires no_sync to cover both the forward and backward.
                        with maybe_no_sync(model, synchronize):
                            with distributed_phase(checkpoint_group, "OPSD loss generation"):
                                # One generator resume runs no-grad rollout/commit and
                                # the checkpointed student with shared LoRA weights.
                                # AMP's process-wide weight cache would make the
                                # original student forward and its isolated backward
                                # recomputation save different tensors.
                                try:
                                    with training_autocast(cache_enabled=False):
                                        loss = next(losses)
                                except StopIteration:
                                    raise RuntimeError(
                                        f"OPSD emitted {term - 1} losses, expected "
                                        f"{terms}, accumulation={accumulation_step + 1}/"
                                        f"{args.gradient_accumulation}"
                                    ) from None
                            detached_loss = loss.detach()
                            require_finite(
                                {"loss": detached_loss},
                                "OPSD loss before backward "
                                f"(accumulation={accumulation_step + 1}/"
                                f"{args.gradient_accumulation}, term={term})",
                                optimizers=(optimizer,),
                            )
                            (loss / accumulated_terms).backward()
                        loss_sum += detached_loss / accumulated_terms
                        del detached_loss, loss
                    with distributed_phase(checkpoint_group, "OPSD loss finalization"):
                        try:
                            with training_autocast(cache_enabled=False):
                                next(losses)
                        except StopIteration:
                            pass
                        else:
                            raise RuntimeError(
                                f"OPSD emitted more than {terms} losses, accumulation="
                                f"{accumulation_step + 1}/{args.gradient_accumulation}"
                            )
            except BaseException:
                with recipe.student_parameters(model, use_adapters):
                    optimizer.zero_grad(set_to_none=True)
                raise
            del losses, sample, contexts, rows
            data_pass, next_batch_index = next_training_position(data_pass, next_batch_index, loader_info.micro_batches)
            if next_batch_index == 0:
                batch_iterator = None
        mean_loss = loss_sum
        if world_size > 1:
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss.div_(world_size)
        grad_norm = clip_grad_norm(model, args.max_grad_norm)
        require_finite(
            {"loss": mean_loss, "grad_norm": grad_norm},
            "OPSD step before optimizer update", optimizers=(optimizer,),
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        step_optimizer(optimizer, scheduler, checkpoint_group, "OPSD optimizer update")
        distributed_call(
            lambda: recipe.update_teacher(
                model, teacher, args.ema_decay, use_adapters=use_adapters,
            ),
            checkpoint_group, "OPSD EMA update",
        )
        global_step += 1
        if progress is not None:
            progress.update(1)
        if global_step == 1 or global_step % args.log_every == 0:
            distributed_call(
                lambda: tracker.log(
                    {
                        "train/loss": mean_loss.item(),
                        "train/grad_norm": float(grad_norm),
                        "train/learning_rate": learning_rate,
                        "time/elapsed_seconds": time.monotonic() - start_time,
                    },
                    step=global_step,
                ) if tracker is not None else None,
                checkpoint_group,
                "OPSD metric logging",
            )
        if global_step % args.save_every == 0:
            save(
                step=global_step, data_pass=data_pass,
                batch_index=next_batch_index,
            )
            last_saved = global_step
    if last_saved != global_step:
        save(
            step=global_step, data_pass=data_pass,
            batch_index=next_batch_index,
        )
    distributed_call(
        tracker.finish if tracker is not None else lambda: None,
        checkpoint_group,
        "OPSD tracker shutdown",
    )

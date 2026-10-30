"""Shared distributed engine for video training recipes."""

import time
from contextlib import closing
from functools import partial

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from backbones import load_backend
from dataset import collate_latent_sequences, dataset_metadata, training_group_ids
from utils import inference_rng
from .evaluation import build_checkpoint_evaluator
from .runtime import (
    TRAINER_STATE_NAME,
    ExperimentTracker,
    accumulation_window,
    build_dataloader,
    build_checkpoint_identity,
    build_lr_scheduler,
    clip_grad_norm,
    close_dataloader,
    common_training_identity,
    distributed_call,
    distributed_mean,
    distributed_strategy,
    export_training_model,
    initialize_training_model,
    load_text_encoder,
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
    optimizer,
    scheduler,
    args,
    backend,
    step,
    data_pass,
    batch_index,
    rank,
    world_size,
    device,
    base_checkpoint,
    checkpoint_identity,
    model_config,
    checkpoint_group,
    evaluate=None,
):
    def write_payload(directory):
        export_training_model(
            directory, model, backend, model_config, base_checkpoint,
            checkpoint_identity, args.lora_rank, args.lora_alpha,
            rank, device, checkpoint_group,
        )
        save_trainer_state(
            directory / TRAINER_STATE_NAME, model, optimizer, scheduler,
            rank, world_size, device, step, data_pass, batch_index,
            checkpoint_group,
        )
        return ("adapter_model.pt",) if args.lora_rank else ()

    return write_training_checkpoint(
        args.output_dir, step, rank=rank, world_size=world_size,
        process_group=checkpoint_group, write_payload=write_payload,
        max_checkpoints=args.max_checkpoints, evaluate=evaluate,
    )


def _initialize_model(
    backend, args, config, training_state, rank
):
    model, base = initialize_training_model(
        backend, args.model_dir, args.checkpoint, config, training_state,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_targets=args.lora_target_modules,
        seed=args.seed, lora_checkpoint=args.lora_checkpoint,
        freeze_camera=args.freeze_camera,
        output_dir=args.output_dir, rank=rank,
    )
    torch.manual_seed(args.seed + rank)
    return model, base


def run_training(args, recipe):
    """Run one recipe without embedding objective logic in the engine."""
    backend = load_backend(args.model)
    with training_session(args.seed) as session:
        return _run_training(backend, args, recipe, *session)


def _run_training(backend, args, recipe, rank, world_size, device, checkpoint_group, resources):
    process_group = None

    model_config = distributed_call(
        lambda: recipe.model_config(backend, args),
        checkpoint_group,
        "Model configuration",
    )
    distributed_call(
        lambda: validate_model_assets(
            args.model_dir,
            args.checkpoint,
            args.model,
            data_path=args.data_path,
            config=model_config,
        ),
        checkpoint_group,
        "Model asset validation",
    )
    needs_text_encoder = backend.uses_text_encoder(
        training=True, config=model_config
    )
    full_dataset = distributed_call(
        lambda: resources.enter_context(closing(recipe.dataset(args))),
        checkpoint_group,
        "Training dataset initialization",
    )

    def validate_data():
        recipe.validate_dataset(backend, full_dataset, model_config)
        return dataset_metadata(full_dataset)

    metadata = distributed_call(
        validate_data,
        checkpoint_group,
        "Training dataset validation",
    )

    def initialize_data():
        if args.num_samples > len(full_dataset):
            raise ValueError(
                f"--num-samples={args.num_samples} exceeds dataset size "
                f"{len(full_dataset)}"
            )
        selected = Subset(full_dataset, range(args.num_samples))
        group_ids = training_group_ids(full_dataset, backend, selected.indices)
        loader, sampler, info = build_dataloader(
            selected,
            args.batch_size,
            args.gradient_accumulation,
            args.num_workers,
            args.seed,
            rank,
            world_size,
            collate_fn=collate_latent_sequences,
            group_ids=group_ids if len(set(group_ids)) > 1 else None,
        )
        resources.callback(close_dataloader, loader)
        return selected, loader, sampler, info

    dataset, dataloader, sampler, loader_info = distributed_call(
        initialize_data,
        checkpoint_group,
        "Training dataloader initialization",
    )
    def initialize_noise_schedule():
        schedule = recipe.schedule(backend, device, model_config, args)
        return schedule, schedule.checkpoint_identity()

    noise_schedule, schedule_identity = distributed_call(
        initialize_noise_schedule, checkpoint_group, "Training noise schedule initialization",
    )
    training_config = recipe.training_config(args, metadata)
    training_config.update(common_training_identity(args, backend))
    training_config["noise_schedule"] = schedule_identity
    if sampler.groups is not None:
        training_config["batch_groups"] = [list(group) for group in sampler.groups]
    training_config["dataset"] = distributed_call(
        full_dataset.checkpoint_identity,
        checkpoint_group,
        "Training dataset identity",
    )
    checkpoint_identity = build_checkpoint_identity(
        args.model,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_target_modules,
        training_config,
        freeze_camera=args.freeze_camera,
    )
    training_state = load_training_state(
        args.checkpoint, args.checkpoint_mode, rank, checkpoint_group
    )
    if training_state is not None:
        distributed_call(
            lambda: validate_training_state(
                training_state,
                checkpoint_identity,
                model_config,
                backend.checkpoint_config(args.checkpoint),
                world_size,
                args.max_train_steps,
                args.checkpoint,
                loader_info,
            ),
            checkpoint_group,
            "Training checkpoint validation",
        )
    distributed_call(
        lambda: args.output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0 else None,
        checkpoint_group,
        "Training output directory creation",
    )

    effective_batch = args.batch_size * args.gradient_accumulation * world_size
    if rank == 0:
        fields = " ".join(
            f"{name}={value}" for name, value in metadata.items()
            if name != "samples"
        )
        print(
            f"dataset_samples={len(full_dataset)} samples={len(dataset)} "
            f"{fields} "
            f"world_size={world_size} "
            f"micro_batch={args.batch_size} effective_batch={effective_batch} "
            f"batch_groups={1 if sampler.groups is None else len(sampler.groups)} "
            f"samples_per_data_pass={loader_info.samples_per_pass} "
            f"repeated_per_data_pass="
            f"{loader_info.repeated_samples_per_pass} "
            f"steps_per_data_pass={loader_info.steps_per_pass} "
            f"max_train_steps={args.max_train_steps} "
            f"checkpoint_mode={args.checkpoint_mode} "
            f"resuming={training_state is not None}",
            flush=True,
        )
        if args.checkpoint_mode == "resume" and training_state is None:
            print(
                "No trainer.pt beside the checkpoint; starting from weights "
                "at step 0",
                flush=True,
            )
        tracker_metadata = {
            name: value for name, value in metadata.items()
            if name != "samples"
        }
        tracker_settings = tracker_config(
            args,
            dataset_samples=len(full_dataset),
            **tracker_metadata,
            world_size=world_size,
            effective_batch=effective_batch,
            batch_groups=1 if sampler.groups is None else len(sampler.groups),
            samples_per_data_pass=loader_info.samples_per_pass,
            repeated_samples_per_data_pass=(
                loader_info.repeated_samples_per_pass
            ),
            steps_per_data_pass=loader_info.steps_per_pass,
            training_mode=recipe.name,
        )
    tracker = distributed_call(
        lambda: resources.enter_context(ExperimentTracker(args, tracker_settings)) if rank == 0 else None,
        checkpoint_group,
        "Experiment tracker initialization",
    )

    text_embeddings = None
    if args.precompute_text_embeddings and needs_text_encoder:
        prompt_groups = distributed_call(
            lambda: [full_dataset.get_prompts(index) for index in range(args.num_samples)],
            checkpoint_group, 'Training prompt collection',
        )
        full_dataset.close()
        text_embeddings = precompute_text_embeddings(
            args.model_dir,
            prompt_groups,
            args.batch_size,
            args.output_dir,
            device,
            rank,
            world_size,
            list(iter(sampler)),
            process_group=process_group,
            error_group=checkpoint_group,
            model=args.model,
        )

    model, base_checkpoint = distributed_call(
        lambda: _initialize_model(
            backend, args, model_config, training_state, rank
        ),
        checkpoint_group,
        "Model initialization",
    )
    if rank == 0:
        if training_state is not None:
            message = "Loaded resumable model weights"
            if args.lora_rank:
                message += " from adapter_model.pt"
            print(message, flush=True)
        if training_state is None and args.lora_checkpoint is not None:
            print(
                f"Loaded initial LoRA checkpoint: {args.lora_checkpoint}",
                flush=True,
            )
        trainable, total = parameter_counts(model)
        print(
            f"trainable_parameters={trainable:,} total_parameters={total:,} "
            f"trainable_percent={100.0 * trainable / total:.4f}",
            flush=True,
        )
    model = wrap_training_model(
        model,
        device,
        parameter_efficient=bool(args.lora_rank),
        transformer_layer_cls=backend.transformer_layer(),
        process_group=process_group,
    )
    if rank == 0:
        print(
            f"distributed_strategy={distributed_strategy(model)}",
            flush=True,
        )

    text_encoder = None
    if text_embeddings is None and needs_text_encoder:
        with inference_rng(device=device):
            text_encoder = distributed_call(
                lambda: load_text_encoder(
                    args.model_dir, device, process_group=process_group, model=args.model,
                ), checkpoint_group, "Text encoder initialization",
            )

    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)
    if training_state is None:
        global_step = data_pass = resume_batch_index = 0
    else:
        global_step, data_pass, resume_batch_index = restore_training_state(
            model,
            optimizer,
            scheduler,
            training_state,
            rank,
            device,
            args.checkpoint.parent,
            checkpoint_group,
        )
        del training_state
    optimizer.zero_grad(set_to_none=True)
    recipe.reset_metrics(device)

    start_time = time.monotonic()
    last_saved_step = global_step
    next_data_pass = data_pass
    next_batch_index = resume_batch_index
    evaluate = distributed_call(
        lambda: build_checkpoint_evaluator(
            args, resources=resources, model=model, backend=backend, device=device, rank=rank,
            world_size=world_size, process_group=checkpoint_group, tracker=tracker,
        ), checkpoint_group, "Checkpoint evaluator initialization",
    )
    save = partial(
        _save_checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
        args=args, backend=backend, rank=rank, world_size=world_size,
        device=device, base_checkpoint=base_checkpoint,
        checkpoint_identity=checkpoint_identity, model_config=model_config,
        checkpoint_group=checkpoint_group, evaluate=evaluate,
    )
    progress = None
    if rank == 0:
        progress = tqdm(
            total=args.max_train_steps,
            initial=global_step,
            desc=recipe.progress_description,
            bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} steps",
            dynamic_ncols=True,
        )
        resources.callback(progress.close)

    while global_step < args.max_train_steps:
        first_batch = resume_batch_index
        batch_iterator = distributed_call(
            lambda: start_training_pass(
                dataloader, sampler, data_pass=data_pass,
                batch_index=first_batch, batch_size=args.batch_size,
                text_stores=(text_embeddings,),
            ), checkpoint_group, "Training data pass",
        )
        for batch_index in range(first_batch, loader_info.micro_batches):
            batch = distributed_call(
                lambda: next(batch_iterator), checkpoint_group, "Training batch loading",
            )
            update_now, window_samples = accumulation_window(
                batch_index,
                loader_info,
                args.batch_size,
                args.gradient_accumulation,
            )
            def encode_contexts():
                if not needs_text_encoder:
                    return None, None
                count = len(batch["prompt"])
                switched = "teacher_prompt_mask" in batch
                prompts = list(batch["prompt"])
                if switched:
                    prompts.extend(batch["teacher_prompt"])
                # Sharded T5 must execute one forward on every rank, even
                # when only some local batches contain prompt switches.
                encoded = backend.encode_context(
                    prompts, device, text_encoder, text_embeddings, batch_index,
                )
                return encoded[:count], encoded[count:] if switched else None

            context, teacher_context = distributed_call(
                encode_contexts, checkpoint_group, "Training context encoding",
            )

            def forward():
                with training_autocast():
                    return recipe.forward(
                        backend,
                        model,
                        batch,
                        context,
                        teacher_context,
                        device,
                        noise_schedule,
                        model_config,
                    )

            with maybe_no_sync(model, update_now):
                per_sample_loss = distributed_call(
                    forward, checkpoint_group, "Training forward",
                )
                require_finite(
                    {"per_sample_loss": per_sample_loss},
                    "forward loss before backward", optimizers=(optimizer,),
                    process_group=process_group,
                )
                (per_sample_loss.sum() / window_samples).backward()
            recipe.accumulate(per_sample_loss)
            del per_sample_loss
            if not update_now:
                continue

            metrics = {
                name: distributed_mean(
                    value, window_samples, world_size, process_group
                ).item()
                for name, value in recipe.pop_metric_sums().items()
            }
            grad_norm = clip_grad_norm(model, args.max_grad_norm)
            require_finite(
                {"grad_norm": grad_norm, **metrics},
                "training step before optimizer update", optimizers=(optimizer,),
                process_group=process_group,
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            step_optimizer(optimizer, scheduler, checkpoint_group, "Training optimizer update")
            global_step += 1
            next_data_pass, next_batch_index = next_training_position(
                data_pass, batch_index, loader_info.micro_batches
            )
            if progress is not None:
                progress.update(1)
            if global_step == 1 or global_step % args.log_every == 0:
                distributed_call(
                    lambda: tracker.log(
                        {
                            **metrics,
                            "train/grad_norm": float(grad_norm),
                            "train/learning_rate": learning_rate,
                            "time/elapsed_seconds": (
                                time.monotonic() - start_time
                            ),
                        },
                        step=global_step,
                    ) if tracker is not None else None,
                    checkpoint_group,
                    "Experiment metric logging",
                )
            if global_step % args.save_every == 0:
                save(
                    step=global_step, data_pass=next_data_pass,
                    batch_index=next_batch_index,
                )
                last_saved_step = global_step
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
        resume_batch_index = 0
        data_pass += 1

    if last_saved_step != global_step:
        save(
            step=global_step, data_pass=next_data_pass,
            batch_index=next_batch_index,
        )
    distributed_call(
        tracker.finish if tracker is not None else lambda: None,
        checkpoint_group,
        "Experiment tracker shutdown",
    )

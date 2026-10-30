"""Distributed three-model training engine for the generic Self Forcing DMD recipe."""

import time
from contextlib import closing
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Subset
from tqdm import tqdm

from backbones import load_backend
from scheduler import sampling_schedule_manifest
from backbones.base import DMDGenerator
from dataset import LatentSequenceDataset, validate_dataset_fields, validate_training_dataset

from .evaluation import build_checkpoint_evaluator
from .runtime import (
    ADAPTER_MODEL_NAME,
    TRAINER_STATE_NAME,
    ExperimentTracker,
    build_checkpoint_identity,
    build_dataloader,
    build_lr_scheduler,
    clip_grad_norm,
    close_dataloader,
    common_training_identity,
    configure_runtime,
    distributed_call,
    distributed_phase,
    distributed_strategy,
    export_training_model,
    initialize_training_model,
    load_checkpoint_config,
    load_training_state,
    maybe_no_sync,
    next_training_position,
    optimizer_parameter_groups,
    parameter_counts,
    precompute_text_embeddings,
    require_finite,
    restore_training_state,
    retain_training_base,
    rng_state_path,
    save_torch_state,
    save_trainer_state,
    start_training_pass,
    step_optimizer,
    tracker_config,
    training_autocast,
    training_session,
    validate_model_assets,
    validate_training_state,
    wrap_fsdp,
    wrap_training_model,
    write_training_checkpoint,
)


def _checkpoint(model, optimizer, lr_scheduler, backend, rank_lora, alpha,
                base, identity, config, directory, step, position, rank,
                world_size, device, group):
    export_training_model(directory, model, backend, config, base, identity,
                          rank_lora, alpha, rank, device, group)
    save_trainer_state(
        directory / TRAINER_STATE_NAME, model, optimizer, lr_scheduler,
        rank, world_size, device, step, *position, group,
    )


def run_dmd_training(args, recipe):
    with training_session(args.seed) as session:
        return _run(args, recipe, *session)


def _run(args, recipe, rank, world_size, device, group, resources):
    resume = load_training_state(args.checkpoint, args.checkpoint_mode, rank, group)
    fake_resume_path = args.checkpoint.parent / "fake_score" / "model.pt"
    fake_resume = (
        load_training_state(fake_resume_path, args.checkpoint_mode, rank, group)
        if resume is not None else None
    )
    if resume is not None and fake_resume is None:
        raise FileNotFoundError(f"DMD resume needs fake score trainer at {fake_resume_path}")
    if resume is not None and any(resume[field] != fake_resume[field]
                                  for field in ("global_step", "data_pass", "batch_index")):
        raise ValueError("DMD generator and fake score resume positions differ")
    fake_checkpoint = fake_resume_path if fake_resume is not None else args.fake_checkpoint

    def inputs():
        backend = load_backend(args.model)
        if not isinstance(backend, DMDGenerator):
            raise TypeError(f"{type(backend).__name__} cannot run a DMD rollout")
        real_backend = load_backend(args.real_model)
        fake_backend = load_backend(args.fake_model)
        for role, selected_backend in (
            ("real", real_backend), ("fake", fake_backend),
        ):
            if not selected_backend.supports_dmd_score:
                raise TypeError(
                    f"{role}_model={type(selected_backend).__name__} does not "
                    "provide a full-context DMD forward"
                )
            if selected_backend.latent_shape[0] != backend.latent_shape[0]:
                raise ValueError(
                    f"{role} score uses {selected_backend.latent_shape[0]} latent channels; "
                    f"generator {type(backend).__name__} uses {backend.latent_shape[0]}"
                )
        initialization = load_checkpoint_config(args.checkpoint)["training"] if resume is not None else {}
        real_checkpoint = args.real_checkpoint
        if initialization.get("real_checkpoint_source") == str(args.real_checkpoint.absolute()):
            real_checkpoint = Path(initialization["real_checkpoint"])
        fake_origin = initialization["fake_checkpoint"] if resume is not None else str(args.fake_checkpoint.resolve())
        model_config = backend.model_config(args)
        real_config = real_backend.checkpoint_config(real_checkpoint) or {}
        fake_config = fake_backend.checkpoint_config(fake_checkpoint) or {}
        validate_model_assets(args.model_dir, args.checkpoint, args.model,
                              config=model_config)
        real_backend.validate(
            args.real_model_dir, real_checkpoint, config=real_config,
        )
        fake_backend.validate(
            args.fake_model_dir, fake_checkpoint, config=fake_config,
        )
        dataset = resources.enter_context(closing(
            LatentSequenceDataset(args.data_path, load_latent=False),
        ))
        validate_training_dataset(dataset, backend)
        for score_backend in (real_backend, fake_backend):
            validate_dataset_fields(dataset, score_backend)
        if args.num_samples > len(dataset):
            raise ValueError(f"num_samples={args.num_samples} exceeds dataset size={len(dataset)}")
        selected = Subset(dataset, range(args.num_samples))
        switching = [i for i in selected.indices if len(dataset.get_prompts(i)) != 1]
        if switching:
            raise ValueError(
                f"DMD requires one prompt per generated video; selected rows {switching} "
                "contain a prompt switch"
            )
        layouts = [backend.dmd_layout(dataset.condition_lengths[i], dataset.lengths[i])
                   for i in selected.indices]
        prompts = [dataset.get_prompt(i) for i in selected.indices]
        return (backend, real_backend, fake_backend, model_config, real_config,
                fake_config, dataset, selected, layouts, prompts, real_checkpoint,
                fake_origin, initialization.get("real_checkpoint_source"))

    (backend, real_backend, fake_backend, model_config, real_config,
     fake_config, dataset, selected, layouts, prompts, real_checkpoint,
     fake_origin, real_origin) = distributed_call(
         inputs, group, "DMD input initialization",
     )
    dataloader, data_sampler, loader_info = distributed_call(
        lambda: build_dataloader(
            selected, args.batch_size, args.gradient_accumulation, args.num_workers,
            args.seed, rank, world_size, collate_fn=list, group_ids=layouts,
        ), group, "DMD data loader initialization",
    )
    resources.callback(close_dataloader, dataloader)
    distributed_call(
        lambda: args.output_dir.mkdir(parents=True, exist_ok=True) if rank == 0 else None,
        group, "DMD output directory creation",
    )
    def initialize_sampler():
        schedule = backend.dmd_schedule(device, args.sampling_steps, sampling_mode=args.sampling_mode)
        return schedule, sampling_schedule_manifest(schedule)

    denoising, sampling_schedule = distributed_call(
        initialize_sampler, group, "DMD sampler initialization",
    )
    identity_fields = {
        **common_training_identity(args, backend),
        "identity_version": 3,
        "critic_learning_rate": args.critic_learning_rate,
        "negative_prompt": args.negative_prompt,
        "sampling_mode": args.sampling_mode,
        "sampling_schedule": sampling_schedule,
        "score_models": {"real": real_backend.training_identity(),
                         "fake": fake_backend.training_identity()},
        "recipe": recipe.name, "generator": args.model,
        "data": distributed_call(dataset.checkpoint_identity, group, "DMD dataset identity"),
        "samples": args.num_samples,
        "layouts": [list(layout) for layout in layouts],
        "steps": args.sampling_steps if isinstance(args.sampling_steps, int) else list(args.sampling_steps),
        "score_shift": args.score_timestep_shift,
        "real_model": args.real_model, "fake_model": args.fake_model,
        "real_checkpoint": str(real_checkpoint.resolve()),
        # Resume loads the trained fake score; its initializer is provenance.
        "fake_checkpoint": fake_origin,
        "real_model_dir": str(args.real_model_dir.resolve()),
        "fake_model_dir": str(args.fake_model_dir.resolve()),
        "gen_ratio": args.generator_update_ratio,
        "gradient_frames": args.gradient_frames, "last_step_only": args.last_step_only,
        "real_cfg": args.real_guidance_scale, "fake_cfg": args.fake_guidance_scale,
        "ema_decay": args.ema_decay, "ema_start_step": args.ema_start_step,
        "worldkv": None,
        "sink_frames": args.sink_frames,
    }
    if real_origin is not None:
        identity_fields["real_checkpoint_source"] = str(args.real_checkpoint.absolute())
    generator_identity = build_checkpoint_identity(
        args.model, args.lora_rank, args.lora_alpha, args.lora_dropout,
        args.lora_target_modules, identity_fields,
        freeze_camera=args.freeze_camera,
    )
    fake_identity = build_checkpoint_identity(
        args.fake_model, args.fake_lora_rank, args.fake_lora_alpha,
        args.lora_dropout, args.lora_target_modules, identity_fields,
        freeze_camera=args.freeze_camera,
    )
    if resume is not None:
        def validate_resume():
            for state, identity, config, selected_backend, checkpoint in (
                (resume, generator_identity, model_config, backend, args.checkpoint),
                (fake_resume, fake_identity, fake_config, fake_backend, fake_resume_path),
            ):
                step = state['global_step']
                # Each critic update consumes a window, and generator updates
                # consume their own windows at steps 0, ratio, 2*ratio, ... .
                data_steps = step + (step + args.generator_update_ratio - 1) // args.generator_update_ratio
                required = None
                if state is resume and args.ema_decay and step >= args.ema_start_step:
                    required = ((ADAPTER_MODEL_NAME, "generator_ema.pt") if identity["lora"]
                                else ("generator_ema.pt",))
                validate_training_state(
                    state, identity, config, selected_backend.checkpoint_config(checkpoint),
                    world_size, args.max_train_steps, checkpoint, loader_info,
                    data_steps=data_steps, required_artifacts=required,
                )
        distributed_call(
            validate_resume, group, "DMD resume validation",
        )

    text_sources = {}
    for role, model_name, model_dir, guidance in (
        ("generator", args.model, args.model_dir, 0.0),
        ("real", args.real_model, args.real_model_dir, args.real_guidance_scale),
        ("fake", args.fake_model, args.fake_model_dir, args.fake_guidance_scale),
    ):
        key = (model_name, str(model_dir.resolve()))
        role, model_dir, negative = text_sources.get(key, (role, model_dir, False))
        text_sources[key] = (role, model_dir, negative or bool(guidance))
    text_stores = {}
    sample_indices = list(iter(data_sampler))
    for key, (role, model_dir, negative) in text_sources.items():
        prompt_groups = [(prompt, args.negative_prompt) if negative else (prompt,) for prompt in prompts]
        text_stores[key] = precompute_text_embeddings(
            model_dir, prompt_groups, args.batch_size,
            args.output_dir / f"{role}_context", device, rank, world_size,
            sample_indices, model=key[0], error_group=group,
        )
    generator_texts = text_stores[(args.model, str(args.model_dir.resolve()))]
    real_texts = text_stores[(
        args.real_model, str(args.real_model_dir.resolve()),
    )]
    fake_texts = text_stores[(
        args.fake_model, str(args.fake_model_dir.resolve()),
    )]

    def build_models():
        generator, source = initialize_training_model(
            backend, args.model_dir, args.checkpoint, model_config, resume,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, lora_targets=args.lora_target_modules,
            seed=args.seed, lora_checkpoint=args.lora_checkpoint, eval_mode=True,
            freeze_camera=args.freeze_camera,
            output_dir=args.output_dir, rank=rank,
        )

        real = real_backend.build(args.real_model_dir, real_config)
        real_backend.load_weights(real, real_checkpoint)
        real.requires_grad_(False).eval()
        real_base = retain_training_base(
            real_backend, real, real_checkpoint, real_config, args.output_dir, rank,
        )
        fake, fake_source = initialize_training_model(
            fake_backend, args.fake_model_dir,
            fake_checkpoint,
            fake_config, fake_resume, lora_rank=args.fake_lora_rank,
            lora_alpha=args.fake_lora_alpha, lora_dropout=args.lora_dropout,
            lora_targets=args.lora_target_modules, seed=args.seed + 1,
            eval_mode=True,
            freeze_camera=args.freeze_camera,
            output_dir=args.output_dir, rank=rank,
        )
        return generator, real, fake, source, fake_source, real_base

    generator, real, fake, base_generator, base_fake, real_base = distributed_call(
        build_models, group, "DMD three-model initialization",
    )
    if real_base != real_checkpoint:
        for identity in (generator_identity, fake_identity):
            identity["training"].update(
                real_checkpoint=str(real_base.resolve()),
                real_checkpoint_source=str(args.real_checkpoint.absolute()),
            )
    generator = wrap_training_model(
        generator, device, parameter_efficient=bool(args.lora_rank),
        transformer_layer_cls=backend.transformer_layer(),
    )
    fake = wrap_training_model(
        fake, device, parameter_efficient=bool(args.fake_lora_rank),
        transformer_layer_cls=fake_backend.transformer_layer(),
    )
    generator_strategy = distributed_strategy(generator)
    fake_strategy = distributed_strategy(fake)
    real = wrap_fsdp(
        real, device, transformer_layer_cls=real_backend.transformer_layer(),
    )
    if rank == 0:
        g_trainable, g_total = parameter_counts(generator)
        f_trainable, f_total = parameter_counts(fake)
        print(f"DMD generator trainable={g_trainable:,}/{g_total:,}; "
              f"fake score trainable={f_trainable:,}/{f_total:,}; "
              f"world_size={world_size} samples={len(selected)}", flush=True)
    generator_optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(generator, args.weight_decay),
        lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    fake_optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(fake, args.weight_decay),
        lr=args.critic_learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    generator_lr = build_lr_scheduler(generator_optimizer, args.lr_warmup_steps)
    fake_lr = build_lr_scheduler(fake_optimizer, args.lr_warmup_steps)
    if resume is None:
        configure_runtime(args.seed, rank)
        step = data_pass = batch_index = 0
    else:
        step, data_pass, batch_index = restore_training_state(
            generator, generator_optimizer, generator_lr, resume, rank, device,
            args.checkpoint.parent, group,
        )
        restore_training_state(
            fake, fake_optimizer, fake_lr, fake_resume, rank, device,
            fake_resume_path.parent, group,
        )
    from model import FullModelEMA

    def initialize_ema(checkpoint=None):
        average = FullModelEMA(generator, args.ema_decay)
        if checkpoint is not None:
            average.load_state_dict(torch.load(checkpoint.with_name("generator_ema.pt"),
                                              map_location="cpu", weights_only=True, mmap=True))
        return average

    ema = (
        distributed_call(lambda: initialize_ema(args.checkpoint if resume is not None else None),
                         group, "DMD EMA initialization")
        if args.ema_decay and step >= args.ema_start_step else None
    )
    resume = fake_resume = None
    generator_optimizer.zero_grad(set_to_none=True)
    fake_optimizer.zero_grad(set_to_none=True)
    tracker = distributed_call(
        lambda: resources.enter_context(ExperimentTracker(
            args, tracker_config(
                args, training_mode=recipe.name, samples=len(selected),
                generator=args.model, real_model=args.real_model,
                fake_model=args.fake_model,
                generator_update_ratio=args.generator_update_ratio,
                batch_size=args.batch_size,
                effective_batch_size=args.batch_size * args.gradient_accumulation * world_size,
            ),
        )) if rank == 0 else None,
        group, "DMD experiment tracker",
    )
    evaluate = distributed_call(
        lambda: build_checkpoint_evaluator(
            args, resources=resources, model=generator, backend=backend, device=device, rank=rank,
            world_size=world_size, process_group=group, tracker=tracker,
        ), group, "Checkpoint evaluator initialization",
    )
    progress = tqdm(total=args.max_train_steps, initial=step,
                    desc=recipe.progress_description, dynamic_ncols=True) if rank == 0 else None
    if progress is not None:
        resources.callback(progress.close)
    iterator = None
    last_saved = step
    started = time.monotonic()
    generator_metrics, generator_batches = {}, 0

    def save():
        nonlocal last_saved
        position = (data_pass, batch_index)

        def write_payload(directory):
            _checkpoint(generator, generator_optimizer, generator_lr, backend,
                        args.lora_rank, args.lora_alpha, base_generator,
                        generator_identity, model_config, directory, step, position,
                        rank, world_size, device, group)
            _checkpoint(fake, fake_optimizer, fake_lr, fake_backend,
                        args.fake_lora_rank, args.fake_lora_alpha, base_fake,
                        fake_identity, fake_config, directory / "fake_score", step,
                        position, rank, world_size, device, group)
            distributed_call(
                lambda: save_torch_state(directory / "generator_ema.pt", ema.state_dict())
                if rank == 0 and ema is not None else None,
                group, "DMD EMA checkpoint write",
            )
            return (
                "fake_score/model.pt", "fake_score/config.json", "fake_score/trainer.pt",
                *(rng_state_path(directory / "fake_score", index, world_size)
                  .relative_to(directory).as_posix() for index in range(world_size)),
                *(("fake_score/adapter_model.pt",) if args.fake_lora_rank else ()),
                *(("adapter_model.pt",) if args.lora_rank else ()),
                *(("generator_ema.pt",) if ema is not None else ()),
            )

        write_training_checkpoint(
            args.output_dir, step, rank=rank, world_size=world_size,
            process_group=group, write_payload=write_payload,
            max_checkpoints=args.max_checkpoints, evaluate=evaluate,
        )
        last_saved = step

    while step < args.max_train_steps:
        update_generator = step % args.generator_update_ratio == 0
        # Match Self-Forcing: finish the generator update before drawing a
        # fresh critic batch and rollout. Each phase accumulates independently.
        for train_generator in ((True, False) if update_generator else (False,)):
            active, optimizer, lr, strategy = (
                (generator, generator_optimizer, generator_lr, generator_strategy)
                if train_generator else (fake, fake_optimizer, fake_lr, fake_strategy)
            )
            phase = "DMD generator" if train_generator else "DMD fake-score"
            total = torch.zeros((), device=device)
            for accumulation in range(args.gradient_accumulation):
                if iterator is None:
                    iterator = distributed_call(
                        lambda: start_training_pass(
                            dataloader, data_sampler, data_pass=data_pass,
                            batch_index=batch_index, batch_size=args.batch_size,
                            text_stores=text_stores.values(),
                        ), group, "DMD data pass",
                    )
                rows = distributed_call(
                    lambda current_iterator=iterator: next(current_iterator),
                    group, "DMD batch loading",
                )
                real_contexts = real_negatives = fake_negatives = None
                with distributed_phase(group, "DMD context encoding"):
                    prompts = [row["prompt"] for row in rows]
                    negative_prompts = [args.negative_prompt] * len(rows)
                    generator_contexts = generator_texts.encode(prompts, batch_index)
                    fake_contexts = fake_texts.encode(prompts, batch_index)
                    if train_generator:
                        real_contexts = real_texts.encode(prompts, batch_index)
                        if args.real_guidance_scale:
                            real_negatives = real_texts.encode(negative_prompts, batch_index)
                        if args.fake_guidance_scale:
                            fake_negatives = fake_texts.encode(negative_prompts, batch_index)
                exit_index = recipe.exit_step(
                    len(denoising.timesteps), device, last_step_only=args.last_step_only,
                )
                synchronize = (strategy == "fsdp" or accumulation + 1 == args.gradient_accumulation)
                generated = None
                if not train_generator:
                    with torch.no_grad(), training_autocast(cache_enabled=False):
                        generated = distributed_call(
                            lambda: backend.dmd_rollout(
                                generator, rows, generator_contexts, denoising,
                                exit_index, device, gradient_frames=args.gradient_frames,
                                sink_frames=args.sink_frames,
                            ), group, "DMD fake-score rollout",
                        )
                with maybe_no_sync(active, synchronize), training_autocast(cache_enabled=False):
                    def loss():
                        if train_generator:
                            return recipe.generator_loss(
                                backend.dmd_rollout(
                                    generator, rows, generator_contexts, denoising,
                                    exit_index, device, gradient_frames=args.gradient_frames,
                                    sink_frames=args.sink_frames,
                                ),
                                real, fake, real_backend, fake_backend,
                                real_contexts, real_negatives,
                                fake_contexts, fake_negatives, rows, args, device,
                            )
                        return recipe.critic_loss(
                            generated.detach(), fake, fake_backend, fake_contexts, rows, args, device,
                        )
                    value = distributed_call(loss, group, phase + " loss generation")
                    require_finite(
                        {"loss": value}, phase + " loss before backward",
                        optimizers=(generator_optimizer, fake_optimizer),
                    )
                    (value / args.gradient_accumulation).backward()
                total += value.detach() / args.gradient_accumulation
                if train_generator:
                    for name, metric in {
                        **recipe.gradient_metrics, "train/generator_loss": value.detach(),
                    }.items():
                        generator_metrics[name] = generator_metrics.get(name, 0) + metric
                    generator_batches += 1
                del (generated, value, generator_contexts, real_contexts,
                     real_negatives, fake_contexts, fake_negatives, rows)
                data_pass, batch_index = next_training_position(
                    data_pass, batch_index, loader_info.micro_batches,
                )
                if batch_index == 0:
                    iterator = None
            norm = clip_grad_norm(active, args.max_grad_norm)
            require_finite(
                {"loss": total, "grad_norm": norm}, phase + " before optimizer update",
                optimizers=(generator_optimizer, fake_optimizer),
            )
            step_optimizer(optimizer, lr, group, phase + " optimizer update")
            if train_generator:
                # Each update contains the same number of microbatches. Weight
                # its norm accordingly to average updates with the loss metrics.
                name = "train/generator_grad_norm"
                generator_metrics[name] = (generator_metrics.get(name, 0)
                                           + norm.detach().double() * args.gradient_accumulation)
                if ema is not None:
                    distributed_call(lambda: ema.update(generator), group, "DMD EMA update")
            else:
                fake_total, fake_norm = total, norm
        step += 1
        if ema is None and args.ema_decay and step >= args.ema_start_step:
            ema = distributed_call(initialize_ema, group, "DMD EMA initialization")
        if progress is not None:
            progress.update(1)
        if step == 1 or step % args.log_every == 0 or step == args.max_train_steps:
            if world_size > 1:
                dist.all_reduce(fake_total)
                fake_total /= world_size
            metrics = {
                "train/fake_score_loss": fake_total.item(),
                "train/fake_score_grad_norm": float(fake_norm),
                "train/generator_lr": generator_optimizer.param_groups[0]["lr"],
                "train/fake_score_lr": fake_optimizer.param_groups[0]["lr"],
                "time/elapsed_seconds": time.monotonic() - started,
            }
            if generator_batches:
                values = torch.stack(tuple(generator_metrics.values())) / generator_batches
                if world_size > 1:
                    dist.all_reduce(values)
                    values /= world_size
                metrics.update(zip(generator_metrics, values.tolist()))
                generator_metrics.clear()
                generator_batches = 0

            distributed_call(
                lambda: tracker.log(metrics, step=step) if tracker is not None else None,
                group, "DMD metric logging",
            )
        if step % args.save_every == 0:
            save()
    if last_saved != step:
        save()
    distributed_call(tracker.finish if tracker else lambda: None,
                     group, "DMD tracker shutdown")

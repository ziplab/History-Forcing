# Out-of-Sight World Modeling

Camera-controlled generation and history-guided OPSD training with DreamX.

See the main README for [installation](../README.md#installation), [downloads and inference](../README.md#quick-start), and [data packing, training and W&B setup](../README.md#training).

## Training

The default configuration is [ar_few_step.yaml](../configs/world_model/dreamx/ar_few_step.yaml). Pass `--config` to select another YAML or `--set key=value` to override a setting. `launcher.NPROC_PER_NODE` selects the GPU count.

`history_guidance_scale` controls history guidance, `teacher_kv` selects teacher context, and `num_supervised_denoising_steps` selects supervised denoising steps.

LoRA training uses DDP by default. Set `train.opsd.sharding_strategy: FULL_SHARD` for FSDP backbone sharding. `SHARD_GRAD_OP` and `HYBRID_SHARD` are also supported.

## Custom Inputs

Pass `--parquet path/to/input.parquet` to the inference launcher. Each row contains an initial image, a prompt and camera actions. Camera options and temporal alignment are stored in the Parquet metadata; the launcher packs model inputs before generation.

## Evaluation

Use the launcher's `evaluate` stage with `--dataset playworld_oos` or `--dataset memobench_real`. `--checkpoint` selects a checkpoint; omitting it evaluates the base model. `--all-checkpoints` evaluates saved checkpoints, and `--mode score` scores existing videos.

- [PlayWorld OOS](evaluation/playworld/README.md): out-of-sight tasks with preset camera paths and VQA scoring.
- [MemoBench Real](evaluation/memobench/README.md): quality, identity, geometry and reference-fidelity metrics.

## Model Interfaces

See [Wan and camera conditioning](src/wan/README.md) for backbone and camera-mechanism integration.

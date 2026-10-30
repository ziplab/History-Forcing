# Long-Video Generation

AR teacher fine-tuning, history-guided DMD and long-video evaluation.

See the main README for [installation](../README.md#installation), [downloads and inference](../README.md#quick-start), and [default training and W&B setup](../README.md#training). Run commands from the repository root.

## Modules

- `model/`, `pipeline/` and `trainer/`: AR teacher and DMD training, rollout and checkpointing.
- `long_video/`: long-video inference; see its [inference guide](long_video/README.md) for settings.
- `evaluation/`: generation and scoring for VBench and VBench-Long.

## Configurations

| Configuration | Purpose |
|---|---|
| [dmd_10s_teacher.yaml](../configs/long_video/dmd_10s_teacher.yaml) | History-guided DMD with the fine-tuned AR teacher |
| [dmd_5s_teacher.yaml](../configs/long_video/dmd_5s_teacher.yaml) | History-guided DMD with the published AR teacher |
| [dmd_baseline.yaml](../configs/long_video/dmd_baseline.yaml) | DMD baseline |
| [ar_teacher_10s.yaml](../configs/long_video/ar_teacher_10s.yaml) | AR teacher fine-tuning |
| [inference.yaml](../configs/long_video/inference.yaml) | Recommended MemRoPE settings for better visual quality in demos. |

Select training settings with `--config_path`. [default_config.yaml](../configs/long_video/default_config.yaml) supplies shared defaults; `history_guidance_scale` and `teacher_kv` control history guidance.

## Training

### AR Teacher Fine-Tuning

```bash
bash scripts/long_video/prepare_ultravideo.sh
bash scripts/long_video/train_ar.sh
```

Preparation downloads and encodes UltraVideo-Long clips into training and validation LMDBs. Use `prepare_ultravideo.sh help` for individual stages.

Training launchers accept `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`. Match `total_batch_size` to the effective batch size. Resume by passing a complete checkpoint's saved `config.yaml` to `--config_path`.

## Evaluation

Evaluation on VBench and VBench-Long uses native autoregressive inference.

Prepare the official VBench code and its [metric weights](https://github.com/Vchitect/VBench#usage):

```bash
git clone https://github.com/Vchitect/VBench.git dmd/evaluation/vbench/VBench
```

```bash
bash scripts/long_video/eval_vbench.sh \
  --config configs/long_video/dmd_10s_teacher.yaml \
  --checkpoint models/History-Forcing/dmd/model.pt
bash scripts/long_video/eval_moviegenbench.sh \
  --config configs/long_video/dmd_10s_teacher.yaml \
  --checkpoint models/History-Forcing/dmd/model.pt
```

VBench defaults to quality dimensions; use `--suite semantic` or `--dimensions` for other selections. MovieGenBench uses [MovieGenVideoBench.txt](prompts/MovieGenVideoBench.txt) and six VBench-Long dimensions. Both launchers use EMA by default; `--weights raw` selects raw weights, and `VBENCH_CACHE_DIR` sets the metric cache.

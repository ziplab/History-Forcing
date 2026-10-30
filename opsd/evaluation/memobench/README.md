# MemoBench Real

Evaluation on 164 real scenes with reference videos and camera trajectories. Run from the repository root:

## Setup

In the [shared environment](../../../README.md#installation), install the geometry and segmentation dependencies:

```bash
pip install --no-build-isolation --no-deps \
  git+https://github.com/facebookresearch/detectron2.git \
  git+https://github.com/facebookresearch/map-anything.git \
  git+https://github.com/facebookresearch/sam3.git
```

SAM3 requires access to its Hugging Face repository. Set `HF_TOKEN` before downloading metric weights, or set `SAM3_CHECKPOINT` to an existing local checkpoint.

Download metric weights using the evaluation cache directories:

```bash
HF_HOME=models/.cache/huggingface TORCH_HOME=models/.cache/torch \
  PYTHONPATH=opsd/src:opsd python -m evaluation.memobench.weights
```

## Evaluation

```bash
hf download xzxxntxdy/History-Forcing-Data --repo-type dataset \
  --local-dir dataset/History-Forcing
bash scripts/world_model/run.sh evaluate --dataset memobench_real \
  --checkpoint models/History-Forcing/opsd/model.pt
```

Without `--checkpoint`, the evaluator uses the base model. Use `--all-checkpoints` to evaluate saved checkpoints or `--mode score` to score existing videos.

## Configuration

`evaluation.datasets.memobench_real` sets the input Parquet, reference assets, camera settings and output name. Inputs are downloaded from [History-Forcing-Data](https://huggingface.co/datasets/xzxxntxdy/History-Forcing-Data).

Metric weights use the configured `HF_HOME` and `TORCH_HOME`.

## Metrics

The evaluator computes Visual Quality, Motion Smoothness, Object Identity, Geo3D, Camera Controllability, ImageReward, ORS, PSNR, SSIM and LPIPS. These use the metric implementations and normalization in `upstream/`. VQA is not part of this evaluation.

Samples are distributed across the configured GPUs and merged into `summary.json`. Videos and scores are saved under each base/checkpoint output directory. The training output directory also contains `memobench_comparison.json` and `memobench_comparison.md`.

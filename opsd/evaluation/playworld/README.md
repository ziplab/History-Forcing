# PlayWorld OOS

Evaluation on 43 PlayWorld out-of-sight tasks. The released inputs contain initial images, scene prompts, preset camera trajectories and task scoring metadata.

Run from the repository root:

```bash
hf download xzxxntxdy/History-Forcing-Data --repo-type dataset \
  --local-dir dataset/History-Forcing
export ARK_API_KEY="your_api_key"
bash scripts/world_model/run.sh evaluate --dataset playworld_oos \
  --checkpoint models/History-Forcing/opsd/model.pt
```

Without `--checkpoint`, the evaluator uses the base model. `--mode score` scores existing videos; `--all-checkpoints` evaluates saved checkpoints.

## Configuration

`evaluation.datasets.playworld_oos` sets the prepared data, output name, worker count and VQA provider. The default uses Doubao Seed 2.0 Lite through `ARK_API_KEY`. Scores depend on the selected judge, which is recorded with the results.

The preset evaluator generates the supplied camera path and uses the task's VQA questions, instruction gate and score aggregation. Generation prompts and scoring instructions are stored separately in the dataset. Outputs are saved under the configured base or checkpoint directory, with results in `summary.json`.

## Additional Protocols

`adapter.py` also supports the closed-loop `player` and training-camera `trajectory` protocols:

```bash
PYTHONPATH=opsd/src:opsd python -m evaluation.playworld.adapter preset --help
```

The closed-loop player uses the SANA-WM agent routine to select actions from generated observations. These protocols use their own inputs and output directories.

The benchmark implementation is included in `upstream/`. See [PlayWorld](https://github.com/hku-sail/PlayWorld) for the benchmark and its agent protocols.

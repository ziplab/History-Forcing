Long video generation with DMD and [MemRoPE](https://github.com/YoungRaeKimm/MemRoPE).

Run from the repository root:

```bash
bash scripts/long_video/inference.sh \
    --prompt "A musician strumming the strings of an acoustic guitar, lost in the melody of their song." \
    --output-dir output/demo/long_video
```

Generation settings are in [inference.yaml](../../configs/long_video/inference.yaml): 252 latent frames, 832×480, 16 FPS.
Use `--prompt-file` for a list of prompts and `--seed` to set the random seed.

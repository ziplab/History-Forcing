# Metrics

All PlayWorld evaluation code is organized under this directory:

```text
metrics/
├── vqa/             # Gemini 3.1 Pro Rubric Verifier
├── automatic/       # intentionally empty
├── aggregate.py     # aggregate Rubric Verifier results
└── gemini_client.py # create the Rubric Verifier API client
```

Run the Rubric Verifier from the repository root with:

```bash
./run_vqa_score.sh
```

The `automatic/` directory is intentionally empty. For automatic metrics,
refer to the official implementations of [VBench](https://github.com/Vchitect/VBench),
[Omni-WorldBench](https://github.com/AMAP-ML/Omni-WorldBench), and
[WorldMark](https://github.com/AlayaLab/WorldMark).

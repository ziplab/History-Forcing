# VQA Rubric Verifier

This folder contains the complete, inspectable VQA scoring path:

| File | Responsibility |
| --- | --- |
| `sampling.py` | Decode the complete video into synchronized 10 FPS primary and 0.5 FPS detail contact sheets. |
| `rubric.py` | Build the task-aware 1–5 physical/visual-quality questions, scoring policy, output schema, and VQA prompt. |
| `instruction_gate.py` | Build the separate Pass/Partial/Fail trajectory gate used for GC and applicable IF/OE cases. |
| `score.py` | Load one task, assemble visual evidence, call the Gemini 3.1 Pro Rubric Verifier, parse JSON, and save the result/cache. |

## What the Rubric Verifier receives

For `--mode score`, the Gemini 3.1 Pro Rubric Verifier receives the task-derived rubric, an optional
reference image, and both chronological contact-sheet streams. The primary
stream is used for motion and temporal continuity; the detail stream is used
for identity, texture, material, color, and small artifacts. The judge returns
per-question evidence and scores plus a normalized final score from 1 to 5.

For `--mode gate`, the same video evidence is used without the reference image.
The result is only the instruction-following verdict. The benchmark aggregation
later converts an applicable Fail to a contribution of 1.

## Temporal granularity and evidence compression

Gemini does not receive a newly encoded low-FPS video. The evaluator decodes the
entire source video into two synchronized, chronological image streams and sends
their contact sheets as visual evidence. Keep these settings fixed when comparing
world models: changing the FPS, spatial scale, or JPEG compression changes the
evidence available to the Rubric Verifier and can therefore change the score.

| Evidence stream | Sampling | Cell size | Contact-sheet grid | Sheet size | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary | 10 FPS (one sample every 0.1 s) | 384 x 216 | 5 x 5, 25 frames | 1920 x 1080 | Motion, action timing, and temporal continuity |
| Detail | 0.5 FPS (one sample every 2 s) | 800 x 450 | 2 x 2, 4 frames | 1600 x 900 | Identity, texture, material, color, and small artifacts |

Frames are ordered left-to-right and then top-to-bottom; sheets are submitted in
chronological order. A full primary sheet contains 2.5 seconds of samples, while a
full detail sheet contains 8 seconds of samples. Sampling starts at 0 seconds and
continues to the end of the video, so clips of different duration produce different
numbers of sheets rather than being truncated to a fixed frame count.

Each sampled frame preserves its original aspect ratio, is resized to fit the
cell, and is centered with black padding when necessary. Contact sheets are encoded
as JPEG with quality 70, optimized encoding, and 4:2:0 chroma subsampling. This
compression applies only to the evidence images sent to Gemini; the input video is
never overwritten or recompressed. The optional reference image is sent separately
at its original file encoding and is used only as initial-scene/style context.

There is no fixed target size in MB because JPEG size varies with visual content and
video duration. By default, evidence totaling at most 18 MiB is included inline;
larger evidence uses the Gemini Files API. This transport switch does not alter the
sampling, resolution, or JPEG quality. It can be overridden with
`--evidence-transport` and `--inline-limit-mb` without changing the benchmark
evidence itself.

To inspect the exact evidence before scoring, export the sheets and manifest:

```bash
DRY_RUN=1 ./run_vqa_score.sh \
  --mode score \
  --dataset /path/data.json \
  --task-id GC001 \
  --video /path/video.mp4 \
  --sampling-output-dir /path/sampling_preview \
  --output /path/dry_run.json
```

`sampling_preview/sampling_manifest.json` records the FPS, sampling interval, cell
and sheet sizes, frame count, time range, ordering, and JPEG settings used in that
run.

## Run

Edit the configuration block at the top of `run_vqa_score.sh`, especially:

```bash
api_key="xxx"
dataset="/absolute/path/to/data.json"
task_id="GC001"
video="/absolute/path/to/recording.mp4"
reference_image="/absolute/path/to/GC001.jpg"
```

Then run:

```bash
chmod +x run_vqa_score.sh
./run_vqa_score.sh
```

To validate sampling and prompt generation without an API call:

```bash
DRY_RUN=1 ./run_vqa_score.sh
```

All settings may instead be supplied as uppercase environment variables. For
advanced use, arguments are forwarded directly:

```bash
GEMINI_API_KEY="xxx" ./run_vqa_score.sh --mode score --dataset /path/data.json \
  --task-id GC001 --video /path/video.mp4 --output outputs/GC001.json
```

The key is read only from the process environment by the Python client and is
never written to the result or context JSON.

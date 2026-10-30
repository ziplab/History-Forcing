# PlayWorld: Benchmarking World Models with Agent Players over Long-Horizon Objectives

[Kaixin Ding](https://kxding.github.io/)<sup>1</sup>, [Xi Chen](https://xavierchen34.github.io/)<sup>1</sup>, [Minghong Cai](https://github.com/kxding/PlayWorld)<sup>2</sup>, [Zhiyuan Xu](https://github.com/kxding/PlayWorld)<sup>1</sup>, [Yuxiang Lu](https://github.com/kxding/PlayWorld)<sup>1</sup>, [Yiyang Wang](https://github.com/kxding/PlayWorld)<sup>1</sup>, [Junyi Li](https://github.com/kxding/PlayWorld)<sup>1</sup>, [Shuyang Chen](https://github.com/kxding/PlayWorld)<sup>2</sup>, [Yuan Gao](https://github.com/kxding/PlayWorld)<sup>2</sup>, [Xin Tao](https://www.xtao.website/)<sup>2</sup>, [Pengfei Wan](https://github.com/kxding/PlayWorld)<sup>2</sup>, [Hengshuang Zhao](https://hszhao.github.io/)<sup>1</sup>

<sup>1</sup>The University of Hong Kong &nbsp;&nbsp; <sup>2</sup>Kuaishou Technology, Kling Team

[**Project**](https://kxding.github.io/project/PlayWorld/) | [**Paper**](https://arxiv.org/abs/2608.13552) | [**Dataset**](https://huggingface.co/datasets/jocelynd/playworld-bench) | [**Leaderboard**](https://huggingface.co/spaces/jocelynd/PlayWorld-Leaderboard)

Video generation-based world models simulate future states conditioned on the current observation and user-provided actions, but their interactive nature makes fair comparison difficult: the action sequences needed to accomplish the same long-horizon objective can differ substantially across models.

PlayWorld therefore drives evaluation by objectives rather than predetermined action sequences. A multi-modal **Agent Player** interacts with each world model in a closed loop and adaptively plans actions toward the same high-level objective. The benchmark uses VQA-based metrics to assess geometric consistency, interaction fidelity, insight evolution, and out-of-sight evolution, together with automatic metrics for further evaluation. Experiments on representative world models show that current systems remain unreliable on long-horizon interactive objectives and continue to struggle with key world-model capabilities.

## Repository Structure

```text
PlayWorld/
├── example/              # runnable GC/IF image-task examples
├── Agent_player/
│   ├── your-models/     # template for integrating another model
│   ├── genie3/          # Genie3 player
│   ├── happyoyster/     # HappyOyster player
│   ├── hyworld2/        # HYWorld2 player
│   └── SANA-WM/         # SANA-WM player
├── metrics/
│   ├── vqa/             # Gemini 3.1 Pro Rubric Verifier
│   ├── automatic/       # empty; add or adapt automatic metrics here
│   ├── aggregate.py     # Rubric Verifier result aggregation
│   └── gemini_client.py # Rubric Verifier API client
├── tests/               # metric and client tests
├── run_vqa_score.sh     # root VQA entry point with api_key="xxx"
└── data/                # downloaded separately; ignored by Git
```
<img width="2530" height="1909" alt="teaser" src="https://github.com/user-attachments/assets/1be6eb7e-6db3-4bc8-9b14-d4ca146896bd" />

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/kxding/PlayWorld.git
cd PlayWorld
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Install the browser runtime when evaluating web-based models such as Genie3 or HappyOyster:

```bash
python3 -m playwright install chromium
```
https://github.com/user-attachments/assets/f5e1708a-f4c0-4fd9-ac8f-891222c2ca08

## Data Preparation

Download the complete [PlayWorld dataset](https://huggingface.co/datasets/jocelynd/playworld-bench) into this repository's ignored `data/` directory:

```bash
hf download jocelynd/playworld-bench --type dataset --local-dir data
```

If the dataset is access-restricted, first request access on Hugging Face and run `hf auth login`. This repository does not download or commit the benchmark data automatically.

The resulting layout should be:

```text
data/
├── mapping_manifest.json
├── validate_datasuite.py
├── gc/
│   ├── data.json
│   └── images/
├── if/
│   ├── data.json
│   └── images/
├── insight/
│   ├── data.json
│   └── images/
└── outsight/
    ├── data.json
    └── images/
```
## Inference Protocol

Inference is driven by an **Agent Player**, which consists of two parts:

- **agent model:** the VLM that observes the latest generated frame, reasons about objective progress, and predicts the next action. The paper uses Claude Haiku for every evaluated world model.
- **Agent Interface and Harness:** the code provided in this repository. It prepares the initial frame, long-horizon objective, step count, and recent action history for the agent model; validates its response; routes the selected action to each world model's native interface; captures the next observation; enforces the shared step budget; and records the rollout.

### Inference requirements

Running inference requires both the target world-model interface and an agent
model API:

- **World-model session:** Genie3 requires a dedicated Chrome CDP session
  launched with `Agent_player/genie3/launch_chrome_cdp.sh`. Running this bash
  script opens a Google Chrome window. Sign in to Genie3 manually in that Chrome
  profile before starting a task. HappyOyster and HYWorld2 use the equivalent
  launch scripts in their model directories and likewise require manual login
  in the opened Chrome window.
- **agent model API:** configure the VLM used to make closed-loop decisions.
  The paper uses Claude Haiku. Copy `Agent_player/api_keys.example.sh` to the
  ignored `Agent_player/api_keys.sh`, replace its `xxx` placeholders, or export
  the corresponding environment variables.

```bash
cp Agent_player/api_keys.example.sh Agent_player/api_keys.sh
# Edit api_keys.sh with the Claude Haiku agent model API settings.
Agent_player/genie3/launch_chrome_cdp.sh
```

After the launch script opens Google Chrome, log in to the target world-model
service manually and keep that browser window open while running the task
script.

The Chrome/CDP session provides access to Genie3; the Claude Haiku API drives
the Agent Player. Neither one is used as the evaluation judge.

**Every world model receives the same initial frame and long-horizon objective, never a predetermined action or camera sequence.** The Agent Player therefore adapts its actions independently to the observations returned by each world model.

The shared action vocabulary includes `W/A/S/D`, arrow keys, and `WAIT`. Model adapters translate these Agent Player decisions to each model's native interface. An episode ends when the agent model outputs `FINISH` or reaches the 40-step limit.

The Agent Interface and harness are reusable across agent models. Agent model credentials are read from `Agent_player/api_keys.sh` or environment variables.

`Agent_player/player.py` defines the shared Agent Interface lifecycle. Each model-specific directory provides an Agent Interface harness for observation capture, native action execution, rollout recording, and model-specific browser automation, while `Agent_player/your-models/` is the copyable integration template. Complete harnesses are provided for Genie3, HappyOyster, and HYWorld2; see [Agent_player/README.md](Agent_player/README.md) for setup and credentials.

Run task `GC004` from the repository root:

Genie3:

```bash
Agent_player/genie3/launch_chrome_cdp.sh
GENIE3_DATA_ROOT="$PWD/example" \
GENIE3_DATA_FILES="GC:GC/001.json" \
Agent_player/genie3/run_task.sh GC004
```

HappyOyster:

```bash
Agent_player/happyoyster/launch_chrome_cdp.sh
HAPPYOYSTER_DATA_ROOT="$PWD/example" \
HAPPYOYSTER_DATA_FILES="GC:GC/001.json" \
Agent_player/happyoyster/run_task.sh GC004
```

HYWorld2:

```bash
Agent_player/hyworld2/launch_chrome_cdp.sh
HYWORLD2_DATA_ROOT="$PWD/example" \
Agent_player/hyworld2/run_task.sh GC004
```

Each launch script opens a Google Chrome window. Sign in to the corresponding
world-model service manually in that window, keep it open, and then run the task
script. Account credentials are never stored in this repository.

## Output Directory Structure

Use one directory per model and task. The reusable player writes the following canonical artifacts; model-specific adapters may add native videos, recordings, execution logs, and intermediate observations.

```text
outputs/
└── <model>/
    └── <task_id>/
        ├── before_actions.jpg
        ├── after_actions.jpg
        ├── result.json
        └── <task_video>.mp4

evaluation/
└── <model>/
    ├── <task_id>_score.json
    └── <task_id>_score_context.json
```

Keep generated videos, evaluation outputs, caches, and API credentials outside Git tracking.

## Running Evaluation

### Step 1: VQA Rubric Verifier

PlayWorldEval uses task-conditioned Gemini 3.1 Pro as the **Rubric Verifier**. This scoring role is separate from Agent Player inference. Each video is decoded into synchronized 10 FPS primary and 0.5 FPS detail streams, then verified against the sample-specific weighted 1–5 rubric. Geometry Consistency and applicable Interaction Fidelity/Out-of-sight cases also use a Pass/Partial/Fail instruction-following gate; an applicable `Fail` contributes an effective score of 1.

Evaluation therefore requires a separate **Gemini 3.1 Pro API key**. The Agent
Model credential used during inference, such as the Claude Haiku API key,
cannot replace it. Configure the verifier key through `GEMINI_API_KEY` or the
`api_key="xxx"` placeholder in `run_vqa_score.sh`.

Edit the configuration block at the top of `run_vqa_score.sh`, including the explicit placeholder:

```bash
api_key="xxx"
dataset="$PWD/data/gc/data.json"
task_id="GC002"
video="$PWD/outputs/genie3/GC002/GC002_full_process.mp4"
reference_image="$PWD/data/gc/images/GC002.jpg"
```

Then run:

```bash
chmod +x run_vqa_score.sh
./run_vqa_score.sh
```

Credentials may instead be provided without editing the file:

```bash
GEMINI_API_KEY="xxx" \
DATASET="$PWD/data/gc/data.json" \
TASK_ID=GC002 \
VIDEO="$PWD/outputs/genie3/GC002/GC002_full_process.mp4" \
REFERENCE_IMAGE="$PWD/data/gc/images/GC002.jpg" \
OUTPUT="$PWD/evaluation/genie3/GC002_score.json" \
./run_vqa_score.sh
```

Use `DRY_RUN=1 ./run_vqa_score.sh` to validate sampling and prompt construction without making an API call. The complete scoring path is documented in [metrics/vqa/README.md](metrics/vqa/README.md).

### Step 2: Automatic Metrics

`metrics/automatic/` is intentionally empty. Automatic video quality and controllability metrics can be added by referring to the official implementations from [VBench](https://github.com/Vchitect/VBench), [Omni-WorldBench](https://github.com/AMAP-ML/Omni-WorldBench), and [WorldMark](https://github.com/AlayaLab/WorldMark).

## Leaderboard

Results below follow the latest PlayWorld experiment table. Geometry Consistency (GC), Interaction Fidelity (IF), Insight Evolution (IE), and Out-of-sight Evolution (OE) use the 1–5 scale under the `Fail = 1` instruction-following policy. Overall is the unweighted mean across the four evaluation dimensions. Best results are bolded and second-best results are underlined.

| **Model** | **GC** | **IF** | **IE** | **OE** | **Overall** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Genie 3 | **2.74** | **2.40** | <u>1.51</u> | **1.81** | **2.12** |
| HappyOyster | <u>2.54</u> | 2.15 | 1.47 | <u>1.54</u> | <u>1.92</u> |
| LingBot-World | 2.11 | <u>2.23</u> | 1.33 | 1.43 | 1.78 |
| LingBot-World2 | 2.04 | 2.13 | **1.95** | 1.16 | 1.82 |
| HY-World2 | 2.14 | 2.06 | 1.13 | 1.09 | 1.61 |
| SANA-WM | 1.72 | 1.89 | 1.13 | 1.16 | 1.48 |
| Hunyuan-GameCraft | 1.62 | 1.52 | 1.21 | 1.31 | 1.42 |
| HY-WorldPlay | 1.12 | 1.63 | 1.01 | 1.08 | 1.21 |
| Matrix-Game-3.0 | 1.30 | 1.25 | 1.00 | 1.00 | 1.14 |

For the live and submission-ready table, visit the [PlayWorld Leaderboard](https://huggingface.co/spaces/jocelynd/PlayWorld-Leaderboard).

## License

The code and benchmark assets have different release considerations. Initial images originate from multiple sources, including Pexels and web image search results, so dataset users must follow the source-specific license or redistribution restrictions documented with the dataset. A single permissive license must not be assumed to cover every image asset.

## Citation

If you find PlayWorld useful, please cite:

```bibtex
@article{ding2026playworld,
  title  = {PlayWorld: Benchmarking World Models with Agent Players over Long-Horizon Objectives},
  author = {Ding, Kaixin and Chen, Xi and Cai, Minghong and Xu, Zhiyuan and Lu, Yuxiang and Wang, Yiyang and Li, Junyi and Chen, Shuyang and Gao, Yuan and Tao, Xin and Wan, Pengfei and Zhao, Hengshuang},
  journal = {arXiv preprint arXiv:2608.13552},
  year   = {2026},
  url    = {https://arxiv.org/abs/2608.13552}
}
```

# Agent Player

The inference-time **Agent Player** consists of an **agent model** and an
**Agent Interface**. This directory provides the reusable Agent Interface,
execution harnesses, examples, and model-specific web adapters.

```text
PlayWorld/
├── example/            # runnable GC/IF image-task examples
└── Agent_player/
    ├── player.py           # shared Agent Interface lifecycle and action parser
    ├── api_keys.example.sh # credential placeholders only
    ├── your-models/        # template for integrating another model
    ├── genie3/             # complete Genie3 player
    ├── happyoyster/        # complete HappyOyster player
    ├── hyworld2/           # complete HYWorld2 player
    └── SANA-WM/            # complete SANA-WM player
```

## Agent Player components

### Agent model

The agent model is the VLM that receives the latest generated frame, the
long-horizon objective, the current step, and recent action history, then
returns the next action. The paper uses Claude Haiku as the agent model for all
evaluated world models. The model remains configurable so future VLM/game
agents can use the same interface and harness.

### Agent Interface and Harness

This repository provides the Agent Interface and harness. Together they:

- load the initial frame and long-horizon objective;
- build the per-step agent model context;
- validate and parse actions from the shared `W/A/S/D`, arrow-key, and `WAIT`
  action space;
- route actions to the target world model's native interface;
- capture the latest generated frame for the next closed-loop decision;
- stop on `FINISH` or the shared 40-step budget; and
- save videos, screenshots, action history, logs, and `result.json`.

The web-model harnesses attach to a signed-in Chrome session through CDP and
use Playwright for end-to-end observation and control.

## Add another world model

`player.py` defines the common Agent Interface lifecycle and action handling.
Start from `your-models/player.py`, then implement page selection, initial-frame
and objective submission, world readiness, observation capture, native action
routing, and optionally native-video download.

## Install

```bash
python3 -m pip install -r Agent_player/genie3/requirements.txt
python3 -m playwright install chromium
```

## Agent model credentials

No source credential is included; agent model API settings are read
from environment variables or the ignored local file:

```bash
cp Agent_player/api_keys.example.sh Agent_player/api_keys.sh
# replace xxx values
```

These credentials drive Agent Player decisions during inference. Gemini 3.1
Pro is the Rubric Verifier, not the agent model; its separate evaluation API is
configured through the root `run_vqa_score.sh`.

## Run a bundled example

Launch the corresponding dedicated Chrome session, sign in once, then run a
task from [`example/`](https://github.com/kxding/PlayWorld/tree/main/example).
The following commands use `GC/001.jpg` and `GC/001.json`, whose original task
ID is `GC004`:

```bash
Agent_player/genie3/launch_chrome_cdp.sh
GENIE3_DATA_ROOT="$PWD/example" \
GENIE3_DATA_FILES="GC:GC/001.json" \
Agent_player/genie3/run_task.sh GC004

Agent_player/happyoyster/launch_chrome_cdp.sh
HAPPYOYSTER_DATA_ROOT="$PWD/example" \
HAPPYOYSTER_DATA_FILES="GC:GC/001.json" \
Agent_player/happyoyster/run_task.sh GC004

Agent_player/hyworld2/launch_chrome_cdp.sh
HYWORLD2_DATA_ROOT="$PWD/example" \
Agent_player/hyworld2/run_task.sh GC004
```

The bundled set contains three GC and three IF tasks. It contains no OE sample;
use the complete Hugging Face dataset for OE runs. See `example/README.md` for
the numbered task mapping and category-specific commands.

The three adapters retain the operational implementations migrated from the
local Genie3, HappyOyster, and HYWorld2 Playwright runners. See each model
directory's README for model-specific details.

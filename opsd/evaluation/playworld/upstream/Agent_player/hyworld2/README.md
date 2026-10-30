# HYWorld2 Agent Player Adapter

This directory contains the Agent Interface and Playwright harness migrated
from `worldplay_0622/code/playwright_HYWorld2`. It controls Tencent Hunyuan 3D
WorldPlay through a signed-in Chrome session, captures observations for the
agent model, routes Agent Player decisions to the native interface, and saves
native video, screen recording, screenshots, and `result.json`. The paper uses
Claude Haiku as the agent model.

## Install and credentials

```bash
python3 -m pip install -r Agent_player/hyworld2/requirements.txt
cp Agent_player/api_keys.example.sh Agent_player/api_keys.sh
# replace xxx values; api_keys.sh is ignored by Git
python3 -m playwright install chromium
```

## Run

```bash
Agent_player/hyworld2/launch_chrome_cdp.sh
HYWORLD2_DATA_ROOT="$PWD/example" \
Agent_player/hyworld2/run_task.sh GC004
```

The launch script opens a Google Chrome window. Sign in to HYWorld2 manually in
that window, keep it open, and then run the task script.

This command uses the bundled `GC/001.jpg` and `GC/001.json` example, whose
original task ID is `GC004`. HYWorld2
copies the source image into the run output before applying its upload-size
constraints, so the repository example is not modified.

The default dataset root is `data/` and the default public output root is
`outputs/hyworld2/`. Override them when needed:

```bash
HYWORLD2_DATA_ROOT=/path/to/data \
HYWORLD2_VIDEO_OUTPUT_ROOT=/path/to/output \
Agent_player/hyworld2/run_task.sh GC002
```

Validate task loading and action parsing without opening the page:

```bash
Agent_player/hyworld2/run_task.sh GC002 --dry-run
```

Run all four Hugging Face dataset splits with:

```bash
Agent_player/hyworld2/run_all.sh
```

No account or API credential is stored in this directory. The runner reads the
ignored `Agent_player/api_keys.sh` file or the corresponding environment
variables.

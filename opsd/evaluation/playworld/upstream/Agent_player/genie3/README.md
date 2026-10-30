# Genie3 Agent Player Adapter

The Genie3 `player.py` implements the Agent Interface and Playwright harness. It
attaches to an already signed-in Chrome session, creates the world from the
initial frame, captures observations for the agent model, routes Agent Player
actions to Genie3's native web interface, and records the result. The paper uses
Claude Haiku as the agent model.

```bash
python3 -m pip install -r Agent_player/genie3/requirements.txt
Agent_player/genie3/launch_chrome_cdp.sh
GENIE3_DATA_ROOT="$PWD/example" \
GENIE3_DATA_FILES="GC:GC/001.json" \
Agent_player/genie3/run_task.sh GC004
```

The launch script opens a Google Chrome window. Sign in to Genie3 manually in
that window, keep it open, and then run the task script.

This command uses the bundled `GC/001.jpg` and `GC/001.json` example, whose
original task ID is `GC004`. Defaults
remain `data/` for complete-dataset inputs and `outputs/genie3/` for outputs.
Task artifacts are written directly under `outputs/genie3/<task_id>/`; Genie3
does not create first-person or third-person subdirectories.
Configure the agent model through the ignored `Agent_player/api_keys.sh` file.

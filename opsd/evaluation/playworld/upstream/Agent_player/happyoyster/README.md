# HappyOyster Agent Player Adapter

The HappyOyster `player.py` implements the Agent Interface and Playwright
harness. It attaches to an already signed-in Chrome session, captures
observations for the agent model, routes Agent Player decisions to the native
web interface, and records the rollout. The paper uses Claude Haiku as the
agent model; the implementation also retains preset-only, agent-only, and
combined diagnostic modes.

```bash
python3 -m pip install -r Agent_player/happyoyster/requirements.txt
Agent_player/happyoyster/launch_chrome_cdp.sh
HAPPYOYSTER_DATA_ROOT="$PWD/example" \
HAPPYOYSTER_DATA_FILES="GC:GC/001.json" \
Agent_player/happyoyster/run_task.sh GC004
```

The launch script opens a Google Chrome window. Sign in to HappyOyster manually
in that window, keep it open, and then run the task script.

This command uses the bundled `GC/001.jpg` and `GC/001.json` example, whose
original task ID is `GC004`. Defaults
remain `data/` for complete-dataset inputs and `outputs/happyoyster/` for
outputs. Task artifacts are written directly under
`outputs/happyoyster/<task_id>/`; HappyOyster does not create first-person or
third-person subdirectories. Configure the agent model through the ignored
`Agent_player/api_keys.sh` file.

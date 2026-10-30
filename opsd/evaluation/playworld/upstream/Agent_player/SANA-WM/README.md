# SANA-WM Agent Player Adapter

The SANA-WM player runs PlayWorld tasks with a local SANA-WM checkout and the
same Agent API configuration used by the other bundled players.

## Install and configure

```bash
python3 -m pip install -r Agent_player/SANA-WM/requirements.txt
cp Agent_player/api_keys.example.sh Agent_player/api_keys.sh

export SANA_ROOT=/path/to/Sana
export SANA_CONFIG=/path/to/config.yaml       # optional
export SANA_CHECKPOINT=/path/to/checkpoint    # optional
```

Fill `PLAYER_API_URL`, `PLAYER_API_KEY`, and `PLAYER_MODEL` in
`Agent_player/api_keys.sh`.

## Run

From the `playworld_code` directory:

```bash
Agent_player/SANA-WM/run_task.sh GC001
Agent_player/SANA-WM/run_task.sh IF001
Agent_player/SANA-WM/run_task.sh OE001
```

The default dataset root is the repository's `data/` directory, matching the
other Agent Player adapters. OE tasks use `data/insight/data.json` by default.
To run out-of-sight tasks:

```bash
SANA_WM_OE_SPLIT=outsight Agent_player/SANA-WM/run_task.sh OE014
```

Override the dataset or output location when needed:

```bash
SANA_WM_DATA_ROOT=/path/to/data \
SANA_WM_OUT_ROOT=/path/to/output \
Agent_player/SANA-WM/run_task.sh GC001,GC002
```

Advanced arguments can be passed directly:

```bash
Agent_player/SANA-WM/run_task.sh GC001 \
  --action-source agent \
  --num-chunks 30 \
  --chunk-frames 33
```

Outputs are saved under `outputs/sana_wm/` by default.

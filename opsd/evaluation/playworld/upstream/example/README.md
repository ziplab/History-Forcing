# Agent Player examples

Each numbered JSON is a one-task dataset and references the JPEG with the same
number. Files are grouped by benchmark dimension, and original task IDs are
preserved. The selected seven public examples contain GC and IF tasks only; no
OE task is included in this small example set.

| Dimension | Example pair | Source task |
| --- | --- | --- |
| GC | `GC/001.jpg` + `GC/001.json` | `GC002` |
| GC | `GC/002.jpg` + `GC/002.json` | `GC004` |
| GC | `GC/003.jpg` + `GC/003.json` | `GC005` |
| GC | `GC/004.jpg` + `GC/004.json` | `GC006` |
| IF | `IF/005.jpg` + `IF/005.json` | `IF046` |
| IF | `IF/006.jpg` + `IF/006.json` | `IF032` |
| IF | `IF/007.jpg` + `IF/007.json` | `IF017` |

For example, run the `GC002` pair from the repository root:

```bash
GENIE3_DATA_ROOT="$PWD/example" \
GENIE3_DATA_FILES="GC:GC/001.json" \
Agent_player/genie3/run_task.sh GC002
```

To run an IF example, select its JSON and original ID, for example:

```bash
GENIE3_DATA_ROOT="$PWD/example" \
GENIE3_DATA_FILES="IF:IF/005.json" \
Agent_player/genie3/run_task.sh IF046
```

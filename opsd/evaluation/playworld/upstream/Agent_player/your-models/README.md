# Add Your Model

Use this directory as the template for a new Agent Interface and harness.
Copy `player.py` into a new sibling directory under `Agent_player/`, then
implement page selection, initial-frame/objective submission, readiness
detection, observation capture, native action routing, and optional native-video
download.

All adapters receive the same initial frame and long-horizon objective. The
agent model observes the latest generated frame and adaptively
chooses actions; it must not replay one predetermined action or camera sequence
across models. Keep agent model credentials in environment variables or the
ignored `Agent_player/api_keys.sh`; never place them in source files or result
JSON.

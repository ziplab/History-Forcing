#!/usr/bin/env bash
# Copy to Agent_player/api_keys.sh and keep the real file untracked.

# Agent Model used by Agent Player during inference (for example, Claude Haiku).
player_api_key="${PLAYER_API_KEY:-xxx}"
player_api_url="${PLAYER_API_URL:-https://your-agent-api.example.com/v1}"
player_model="${PLAYER_MODEL:-claude-haiku-4-5-20251001}"
export PLAYER_API_KEY="$player_api_key"
export PLAYER_API_URL="$player_api_url"
export PLAYER_MODEL="$player_model"

# Gemini 3.1 Pro is configured separately in run_vqa_score.sh as the Rubric Verifier.

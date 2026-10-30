#!/usr/bin/env python3
"""Minimal template for adding another web-based world model."""

from pathlib import Path
import sys

from playwright.sync_api import BrowserContext, Page

try:
    from Agent_player.player import Player, PlayerTask
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Agent_player.player import Player, PlayerTask


class YourModelPlayer(Player):
    model_name = "your_model"

    def select_page(self, context: BrowserContext) -> Page:
        raise NotImplementedError("Select or open the signed-in model page")

    def create_world(self, task: PlayerTask, output_dir: Path) -> None:
        raise NotImplementedError("Upload task.image_path and submit task.prompt")

    def wait_until_ready(self, task: PlayerTask, output_dir: Path) -> None:
        raise NotImplementedError("Wait until the world accepts actions")


if __name__ == "__main__":
    raise SystemExit("Copy this template into Agent_player/<your-model>/player.py")

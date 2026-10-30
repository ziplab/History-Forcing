import argparse
import os
import torch.distributed as dist
import wandb

from utils.paths import load_config, resource_path, save_config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")
    parser.add_argument("--checkpoint-mode", choices=("resume", "weights"), default=None)

    args = parser.parse_args()

    from trainer import DiffusionTrainer, ScoreDistillationTrainer

    config = load_config(args.config_path)
    if args.checkpoint_mode is not None:
        config.checkpoint_mode = args.checkpoint_mode
    config.no_save = args.no_save or config.get("no_save", False)
    config.no_visualize = args.no_visualize or config.get("no_visualize", False)
    config.tf = args.tf 
    # get the filename of config_path
    config_name = config.get("config_name") or os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = str(resource_path(args.logdir)) if args.logdir else config.get("logdir", "")
    config.wandb_save_dir = str(resource_path(args.wandb_save_dir)) if args.wandb_save_dir else config.get("wandb_save_dir", "")
    config.disable_wandb = args.disable_wandb or config.get("disable_wandb", False)

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    else:
        raise ValueError(f"Unsupported trainer {config.trainer!r}; use diffusion for AR teacher "
                         "or score_distillation for DMD")
    from utils.checkpoint import _collective_call
    # Each node retains the small run config; complete model checkpoints remain
    # published only by global rank 0 on the existing checkpoint transport.
    _collective_call(
        lambda: save_config(config, os.path.join(config.logdir, "config.yaml"))
        if int(os.environ.get("LOCAL_RANK", "0")) == 0 else None,
        "Save resolved training config")
    trainer.train()

    wandb.finish()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

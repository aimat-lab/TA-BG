import os

import torch
import wandb
from bgflow import BoltzmannGenerator
from torch.optim.lr_scheduler import LRScheduler

from annealed_bg.config.experiment import ExperimentConfig
from annealed_bg.utils.wandb import get_newest_checkpoint_from_wandb_id


def load_checkpoint(
    cfg: ExperimentConfig,
    generator: BoltzmannGenerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler | None,
    warmup_scheduler: LRScheduler | None,
) -> tuple[int, str]:

    checkpoint_path, checkpoint_i = get_newest_checkpoint_from_wandb_id(
        cfg.resume_wandb_id, cfg.checkpoint_i
    )

    starting_i = checkpoint_i + 1

    print(
        "Resuming run with wandb id",
        cfg.resume_wandb_id,
        "from checkpoint at i =",
        checkpoint_i,
        "...",
    )

    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if not cfg.only_run_eval:
        assert (lr_scheduler is None) == (checkpoint["lr_scheduler"] is None)
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if not cfg.only_run_eval:
        assert (warmup_scheduler is None) == (checkpoint["warmup_scheduler"] is None)
    if warmup_scheduler is not None:
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler"])

    return starting_i, checkpoint_path


def save_checkpoint(
    current_i: int,
    generator: BoltzmannGenerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler | None,
    warmup_scheduler: LRScheduler | None,
) -> str:
    save_dir = f"{wandb.run.dir}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "model": generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
            "warmup_scheduler": (
                warmup_scheduler.state_dict() if warmup_scheduler is not None else None
            ),
        },
        f"{save_dir}/checkpoint_{current_i}.pt",
    )

    return save_dir

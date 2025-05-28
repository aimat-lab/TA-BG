import sys
from typing import Literal

import torch
import wandb


def build_resume_python_command() -> str:
    """Build python command that can be used to resume the current experiment in the next job."""

    command = "python "
    for item in sys.argv:
        if "resume_wandb_id" not in item:
            command += '"' + item + '" '
    command = command.strip()
    command += f' "experiment.resume_wandb_id={wandb.run.id}"'

    return command


def set_precision(precision: Literal["double", "single"]):
    if precision == "double":
        torch.set_default_dtype(torch.float64)
    elif precision == "single":
        torch.set_default_dtype(torch.float32)
    else:
        raise ValueError(f"Precision {precision} not supported.")

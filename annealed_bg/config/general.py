from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt


class GeneralConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_workers: PositiveInt  # Workers used for target evals
    precision: Literal["double", "single"]
    wandb_log_every: PositiveInt

from typing import List

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wandb_notes: str
    wandb_group: str | None
    wandb_tags: List[str] | None
    disable_wandb: bool
    wandb_offline: bool

    resume_wandb_id: str | None  # wandb id to resume from
    checkpoint_i: int | None  # Checkpoint index to resume from
    resume_after_in_h: (
        float | None
    )  # Terminate once this time has passed, resume in the next job
    wandb_force_new_experiment: bool  # When resuming, still force a new wandb experiment instead of resuming the old one

    only_run_eval: bool  # No training, only run evaluation
    only_run_eval_include_sampling: bool
    only_run_eval_include_additional_sampling: bool
    only_run_eval_include_NLL: bool
    only_run_eval_include_forward_ESS: bool
    eval_outdir: str | None

    @model_validator(mode="after")
    def check(self) -> Self:
        assert not (
            self.disable_wandb and self.wandb_offline
        ), "Cannot disable wandb and run offline."

        if self.eval_outdir is not None and not self.only_run_eval:
            raise ValueError("eval_outdir is only allowed if only_run_eval is True")

        return self

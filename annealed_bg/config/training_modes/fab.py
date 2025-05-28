from typing import Literal

from pydantic import BaseModel, ConfigDict

from annealed_bg.config.training_modes.ais_base import AISBaseTrainingModeConfig


class FabReplayBufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_updates: int
    min_length: int
    max_length: int
    max_adjust_w_clip: int


class FabTrainingModeConfig(AISBaseTrainingModeConfig):
    mode_name: Literal["fab"]

    adjust_step_size_training: bool
    replay_buffer: FabReplayBufferConfig | None

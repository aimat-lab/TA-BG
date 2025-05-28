from typing import Literal

from annealed_bg.config.training_modes.base import TrainingModeConfig


class ForwardKLDTrainingModeConfig(TrainingModeConfig):
    mode_name: Literal["forward_kld"]

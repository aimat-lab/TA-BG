from typing import Literal

from pydantic import NonNegativeInt

from annealed_bg.config.training_modes.base import TrainingModeConfig


class ReverseKLDTrainingModeConfig(TrainingModeConfig):
    mode_name: Literal["reverse_kld"]

    remove_top_k_energies: NonNegativeInt | None

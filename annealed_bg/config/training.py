from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from annealed_bg.config.lr_schedulers import (
    CosineAnnealingLRSchedulerConfig,
    CosineAnnealingWarmRestartsLRSchedulerConfig,
    StepLRSchedulerConfig,
)
from annealed_bg.config.training_modes.fab import FabTrainingModeConfig
from annealed_bg.config.training_modes.forward_kld import ForwardKLDTrainingModeConfig
from annealed_bg.config.training_modes.reverse_kld import ReverseKLDTrainingModeConfig
from annealed_bg.config.training_modes.reweighting import ReweightingTrainingModeConfig


class EnergyRegularizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy_high: float | None
    energy_max: float | None


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float
    batch_size: PositiveInt
    max_iter: NonNegativeInt

    training_mode: (
        FabTrainingModeConfig
        | ForwardKLDTrainingModeConfig
        | ReverseKLDTrainingModeConfig
        | ReweightingTrainingModeConfig
    ) = Field(discriminator="mode_name")

    max_grad_norm: float | None  # Gradient norm clipping
    warmup_iters: NonNegativeInt | None  # Linear lr warmup iterations
    weight_decay: float | None

    energy_regularizer: (
        EnergyRegularizerConfig | None
    )  # Ignored for non-OpenMM type systems!

    lr_scheduler: (
        CosineAnnealingWarmRestartsLRSchedulerConfig
        | CosineAnnealingLRSchedulerConfig
        | StepLRSchedulerConfig
        | None
    ) = Field(discriminator="scheduler_type")

    # Checkpoint to be loaded into generator before starting training. If "default", the default checkpoint from the system config is used.
    # This can be used to initialize the generator with a pretrained model.
    checkpoint_path: str | None

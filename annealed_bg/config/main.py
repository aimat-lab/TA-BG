from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from annealed_bg.config.checkpointing import CheckpointingConfig
from annealed_bg.config.evaluation import EvaluationConfig
from annealed_bg.config.experiment import ExperimentConfig
from annealed_bg.config.flow import FlowConfig
from annealed_bg.config.general import GeneralConfig
from annealed_bg.config.system import SystemConfig
from annealed_bg.config.training import TrainingConfig
from annealed_bg.config.training_modes.ais_base import AISBaseTrainingModeConfig


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # In case of losses that train at a single temperature, this temperature is specified as "main_temp".
    # In case of the reweighting training mode, which operates on a temperature range, "main_temp" is typically the boundary temperature.
    main_temp: float

    flow: FlowConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    checkpointing: CheckpointingConfig
    system: SystemConfig
    general: GeneralConfig

    experiment: ExperimentConfig

    @model_validator(mode="after")
    def check(self) -> Self:
        if isinstance(self.training.training_mode, AISBaseTrainingModeConfig):
            if self.training.training_mode.AIS_trafo_config is not None:
                for (
                    channel_name,
                    config,
                ) in self.training.training_mode.AIS_trafo_config.items():
                    if config[0]:
                        assert (
                            self.flow.couplings_transform_type == "spline"
                        ), "Unbounding with atanh is only supported when using spline couplings"

            if self.training.training_mode.reject_spline_OOB_samples:
                for (
                    channel_name,
                    config,
                ) in self.training.training_mode.AIS_trafo_config.items():
                    assert (
                        self.flow.couplings_transform_type == "spline"
                    ), "Rejecting OOB samples is only supported when using spline couplings"
        return self

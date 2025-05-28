from pydantic import BaseModel, ConfigDict

from annealed_bg.config.training_modes.base import TrainingModeConfig


class AISAdditionalSamplingEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_p2q: bool  # Whether to also evaluate p^2/q


class AISBaseTrainingModeConfig(TrainingModeConfig):
    n_int_dist: int
    n_outer: int
    n_inner: int
    epsilon: float
    common_epsilon_init_weight: float

    # Trafo applied on the ICs before performing AIS
    # Syntax: {channel_name: (perform unbound atanh trafo?, scaler)}
    AIS_trafo_config: dict[str, tuple[bool, float]] | None

    alpha: float

    additional_sampling_eval: AISAdditionalSamplingEvalConfig

    reject_spline_OOB_samples: bool

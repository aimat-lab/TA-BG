from typing import List, Literal, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from annealed_bg.config.training_modes.base import TrainingModeConfig


# Used for temperature-conditioned flows:
class ContextPreprocessorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_to_max_range: bool
    apply_log: bool


class GrowRangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["linear"]  # Grow range linearly to the left
    min_step: NonNegativeInt  # When to start growing the temperature range
    max_step: NonNegativeInt  # When to stop growing the temperature range


class SupportTDistributionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dist_type: Literal["uniform", "left_bound_of_current_range"]
    one_T_per_batch: bool
    fraction_at_boundary_T: (
        float | None
    )  # Force a fraction of the samples to be at the boundary temperature
    fraction_at_boundary_T_1_before_iteration: (
        NonNegativeInt | None
    )  # Force all samples to be at the boundary temperature before the specified iteration


class ReweightTDistributionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dist_type: Literal["delta_uniform", "delta_uniform_left", "delta_constant_left"]
    max_delta_T: float | None
    one_T_per_batch: bool


##### Define temp sampling strategies #####


class ContinuousTempSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temp_sampling_strategy: Literal["continuous"]

    min_temperature_range: (
        Tuple[float, float] | None
    )  # Only used if grow_range is specified
    grow_range: GrowRangeConfig | None

    support_T_distribution: SupportTDistributionConfig
    reweight_T_distribution: ReweightTDistributionConfig


class SequenceTempSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temp_sampling_strategy: Literal["sequence"]

    # Format for each sequence step:
    # - (sampling_T, reweighting_T, [iterations_per_step], [reinit generator], [n_samples], [resample_to], [change_lr_to])
    # Note: n_samples_per_T and resample_to are only allowed to be specified if a buffer is used!
    sequence: List[
        Tuple[float, float]
        | Tuple[float, float, int]
        | Tuple[float, float, int, bool]
        | Tuple[float, float, int, bool, int]
        | Tuple[float, float, int, bool, int, int]
        | Tuple[float, float, int, bool, int, int, float]
    ]
    iterations_per_step: (
        PositiveInt
        | None  # This default is used if not specified in a sequence step directly (see above)
    )


##########


class BufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activate_buffer_after: int
    update_buffer_every: (
        PositiveInt | None
    )  # "None" only allowed if sequence temp sampling is used; In this case, the buffer is updated before each sequence step.

    buffer_n_samples_per_T: (
        PositiveInt | None
    )  # "None" only allowed if sequence temp sampling is used. Here, the number of samples can be directly specified in the sequence step.
    resample_to: (
        PositiveInt | None
    )  # If this is None, we use the weights for reweighting in the forward KLD loss directly
    clip_top_k_weights_fraction: float | None


class ReweightingTrainingModeConfig(TrainingModeConfig):
    mode_name: Literal["reweighting"]

    max_temperature_range: Tuple[float, float]

    temp_sampling: ContinuousTempSamplingConfig | SequenceTempSamplingConfig = Field(
        discriminator="temp_sampling_strategy"
    )

    context_preprocessor: ContextPreprocessorConfig | None

    buffer: (
        BufferConfig | None
    )  # Buffer not supported for continuous temperature sampling => Needs to be None in this case

    resample_batch_to: PositiveInt | None  # Only used if no buffer is used.
    self_normalize_weights: bool  # Only used if no buffer is used.
    clip_top_k_weights: NonNegativeInt | None  # Only used if no buffer is used.

    @model_validator(mode="after")
    def check(self) -> Self:
        assert (
            self.max_temperature_range[0] <= self.max_temperature_range[1]
        ), "Max temperature range must be increasing"

        if isinstance(self.temp_sampling, ContinuousTempSamplingConfig):
            if self.temp_sampling.min_temperature_range is not None:
                assert (
                    self.temp_sampling.min_temperature_range[0]
                    <= self.temp_sampling.min_temperature_range[1]
                ), "Min temperature range must be increasing"
                assert (
                    self.temp_sampling.min_temperature_range[1]
                    == self.max_temperature_range[1]
                ), "Min temperature range must end at the same temperature as max temperature range"

        if self.buffer is not None:
            assert not isinstance(
                self.temp_sampling, ContinuousTempSamplingConfig
            ), "Buffering is not supported for continuous temperature sampling"

            if self.buffer.update_buffer_every is None:
                assert isinstance(
                    self.temp_sampling, SequenceTempSamplingConfig
                ), "`update_buffer_every` must be set if temp_sampling is not sequence"

            if self.buffer.buffer_n_samples_per_T is None:
                assert isinstance(
                    self.temp_sampling, SequenceTempSamplingConfig
                ), "`buffer_n_samples_per_T` must be set if temp_sampling is not sequence"

        return self

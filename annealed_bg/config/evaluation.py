from typing import List, Tuple

from pydantic import BaseModel, ConfigDict, PositiveInt


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apply_cart_permutation_to_ground_truth_datasets: (
        bool  # Only used for molecular systems
    )

    eval_every: PositiveInt
    eval_samples: PositiveInt

    big_eval_every: PositiveInt
    big_eval_samples: PositiveInt

    # Additional sampling eval is implemented directly in the training mode.
    # We use this, for example, for AIS evaluation.
    additional_sampling_eval_every: PositiveInt
    additional_sampling_eval_samples: PositiveInt

    NLL_every: PositiveInt

    # Overwrite the default temperature pairs to evaluate when sampling from the generator:
    # List of (temperature to sample at (None means no temperature-conditioning), temperature to reweight to (None means no reweighting))
    overwrite_eval_sampling_T_pairs: List[Tuple[float | None, float | None]] | None
    overwrite_eval_NLL_Ts: List[float] | None
    calculate_forward_ESS_for_Ts: List[float]

    skip_initial_eval: bool  # Skip the evaluation at the beginning of training

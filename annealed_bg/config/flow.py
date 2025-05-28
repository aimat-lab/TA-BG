from typing import List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, PositiveInt


class FlowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: List[
        Tuple[str, str, bool, bool]
    ]  # Syntax per list element (layer): ["What to transform", "What to condition on", Also add the reverse?, temperature-aware?]
    min_energy_structure_path: str | None
    torsion_shifts: bool
    use_sobol_prior: bool

    couplings_transform_type: Literal["spline", "rnvp"]
    spline_num_bins: int

    # kwargs for the conditioner networks:
    hidden: List[PositiveInt]
    use_silu_activation: bool  # Instead of relu
    add_skip_connection: bool

    constrain_hydrogen_permutation: bool  # Only used for molecular systems

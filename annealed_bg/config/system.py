from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, ConfigDict


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str

    # Training and validation dataset paths per temperature (str(float(T)):
    data_type: Literal["openmm", "raw"]
    val_data: Dict[str, str]
    train_data: Dict[str, str] | None

    constrain_chirality: bool  # Only used for molecular systems

    marginals_2D: Dict[
        str, List[Tuple[int, int]]
    ]  # For each IC channel: List of 2D marginals used for evaluation (pairs of indices)
    marginals_2D_vmax: float | None

    tica_path_and_selection: Tuple[str, str] | None
    tica_vmax: float | None

    eval_IS_clipping_fraction: float | None

    # Default checkpoint to use if "checkpoint_path" is set to "default".
    # Specified per boundary temperature (str(float(T))):
    default_checkpoint_paths: Dict[str, str] | None

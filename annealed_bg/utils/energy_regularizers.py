import numpy as np
import torch


def lin_log_cut(
    energy: torch.Tensor, energy_high: float, energy_max: float
) -> torch.Tensor:
    # Check whether energy finite
    energy_finite = torch.isfinite(energy)

    # Cap the energy at energy_max
    energy = torch.where(energy < energy_max, energy, energy_max)

    # Make it logarithmic above energy_high and linear below
    energy = torch.where(
        energy < energy_high, energy, torch.log(energy - energy_high + 1) + energy_high
    )

    energy = torch.where(
        energy_finite,
        energy,
        torch.tensor(np.nan, dtype=energy.dtype, device=energy.device),
    )

    return energy

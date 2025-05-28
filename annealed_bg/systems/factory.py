from annealed_bg.config.training import EnergyRegularizerConfig
from annealed_bg.systems.base import System
from annealed_bg.systems.gmm.system import GMMSystem
from annealed_bg.systems.molecular.system import MolecularSystem
from annealed_bg.systems.molecular.utils import openmm_systems


def create_system(
    system_name: str,
    energy_regularizer_cfg: EnergyRegularizerConfig | None,
    n_workers: int,
    system_temp: float,
) -> System:
    if system_name in openmm_systems:
        system = MolecularSystem(
            system_name, energy_regularizer_cfg, n_workers, system_temp
        )
    elif system_name == "2D-GMM":
        system = GMMSystem(system_temp)
    else:
        raise ValueError(f"Unknown system {system_name}")

    return system

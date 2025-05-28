from bgmol.systems import OpenMMSystem
from bgmol.systems.ala2 import AlanineDipeptideImplicit
from bgmol.systems.ala4 import AlanineTetrapeptideImplicit
from bgmol.systems.minipeptides import MiniPeptide

openmm_systems = ["aldp", "tetra", "hexa"]


def get_openmm_system(name: str, n_workers: int, system_temp: float) -> OpenMMSystem:
    if name == "aldp":
        system = AlanineDipeptideImplicit(constraints=None, hydrogenMass=None)
    elif name == "tetra":
        system = AlanineTetrapeptideImplicit(
            root="./input_files/tetra/", download=False
        )
    elif name == "hexa":
        system = MiniPeptide(
            aminoacids="AAAAA",
            download=False,
            root="./input_files/hexa/",
        )
    else:
        raise ValueError(f"Unknown system {name}")

    system.reinitialize_energy_model(
        n_workers=n_workers,
        temperature=system_temp,
    )

    return system

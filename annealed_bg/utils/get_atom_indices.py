from annealed_bg.systems.molecular.system import MolecularSystem
import os


def get_atom_indices_corresponding_to_torsion(
    system_name: str, torsion_index: int
) -> int:

    system = MolecularSystem(
        system_name=system_name,
        energy_regularizer_cfg=None,
        n_workers=1,
        system_temp=300.0,
    )

    all_torsions = system.coordinate_trafo_openmm.torsion_indices

    return all_torsions[torsion_index, :].tolist()


if __name__ == "__main__":
    os.chdir("..")
    system = input("Enter system name:")
    while True:
        torsion_index = int(input("Enter torsion index:"))
        print(
            "Corresponding atom indices:",
            get_atom_indices_corresponding_to_torsion(system, torsion_index),
        )

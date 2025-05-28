import itertools

import numpy as np
import torch
from bgflow import BoltzmannGenerator
from bgflow.nn.flow.sequential import SequentialFlow

from annealed_bg.systems.molecular.system import MolecularSystem
from annealed_bg.utils.dataloading import convert_to_IC


def apply_cart_permutation(
    cart_datasets_to_apply_permutation_to: dict,
    generator: BoltzmannGenerator,
    val_samples_cart: torch.Tensor,
    system: MolecularSystem,
) -> bool:
    """In principle, the energy function of the molecular systems is invariant under permutations of, for example,
    hydrogens in CH3 groups. However, data from MD will not be permutation invariant, but rather follow the permutation
    of the starting structure. Our Boltzmann generators, however, are initialized randomly, so they will acquire
    a random permutation of the hydrogens in CH3 groups. This function will apply the best permutation to the cartesian
    ground truth datasets, matching the permutation of the Boltzmann generator.
    """

    flow_samples_cart = generator.sample(val_samples_cart.shape[0], context=None)
    flow_samples_IC = [
        item.cpu()
        for item in system.coordinate_trafo_openmm.forward(
            flow_samples_cart.cuda(), inverse=False
        )[0:3]
    ]

    permute_hydrogens_of = []
    hydrogens_of_carbons = {}
    for carbon in system.openmm_system.topology.atoms():
        if not carbon.element.name == "carbon":
            continue

        # Get all hydrogens connected to this carbon atom:
        hydrogens = [
            (bond[0] if bond[0].name.startswith("H") else bond[1])
            for bond in system.openmm_system.topology.bonds()
            if (
                (bond[0] == carbon and bond[1].name.startswith("H"))
                or (bond[1] == carbon and bond[0].name.startswith("H"))
            )
        ]

        if len(hydrogens) > 1:
            permute_hydrogens_of.append(carbon)
            hydrogens_of_carbons[carbon] = hydrogens

    permutation_applied = False
    for group_carbon in permute_hydrogens_of:
        hydrogens = hydrogens_of_carbons[group_carbon]
        hydrogens_cart_indices = [hydrogen.index for hydrogen in hydrogens]

        # Determine all permutations of the hydrogens in this group:
        permutations = list(itertools.permutations(hydrogens_cart_indices))
        assert permutations[0] == tuple(hydrogens_cart_indices)

        ##### Determine which IC DOF belong to the atoms in this permutation groups #####

        all_bonds = system.coordinate_trafo_openmm.bond_indices
        relevant_bond_indices = []
        for i, bond in enumerate(all_bonds):
            if any([(index in bond) for index in hydrogens_cart_indices]):
                relevant_bond_indices.append(i)

        all_angles = system.coordinate_trafo_openmm.angle_indices
        relevant_angle_indices = []
        for i, angle in enumerate(all_angles):
            if any([(index in angle) for index in hydrogens_cart_indices]):
                relevant_angle_indices.append(i)

        all_torsions = system.coordinate_trafo_openmm.torsion_indices
        relevant_torsion_indices = []
        for i, torsion in enumerate(all_torsions):
            if any([(index in torsion) for index in hydrogens_cart_indices]):
                relevant_torsion_indices.append(i)

        flow_samples_bonds_mean = torch.mean(
            flow_samples_IC[0][:, relevant_bond_indices], dim=0
        )
        flow_samples_angles_mean = torch.mean(
            flow_samples_IC[1][:, relevant_angle_indices], dim=0
        )
        flow_samples_torsions_mean = torch.mean(
            flow_samples_IC[2][:, relevant_torsion_indices], dim=0
        )

        errors = []
        for permutation in permutations:
            val_samples_perm = val_samples_cart.clone().view(
                val_samples_cart.shape[0], -1, 3
            )
            val_samples_perm[:, hydrogens_cart_indices, :] = val_samples_perm[
                :, torch.tensor(permutation), :
            ]
            val_samples_perm = val_samples_perm.view(val_samples_perm.shape[0], -1)

            val_samples_perm_IC = [
                item.cpu()
                for item in system.coordinate_trafo_openmm.forward(
                    val_samples_perm.cuda(), inverse=False
                )[0:3]
            ]

            val_samples_perm_bonds_mean = torch.mean(
                val_samples_perm_IC[0][:, relevant_bond_indices], dim=0
            )
            val_samples_perm_angles_mean = torch.mean(
                val_samples_perm_IC[1][:, relevant_angle_indices], dim=0
            )
            val_samples_perm_torsions_mean = torch.mean(
                val_samples_perm_IC[2][:, relevant_torsion_indices], dim=0
            )

            errors.append(
                (
                    torch.mean(
                        torch.abs(flow_samples_bonds_mean - val_samples_perm_bonds_mean)
                    )
                    + torch.mean(
                        torch.abs(
                            flow_samples_angles_mean - val_samples_perm_angles_mean
                        )
                    )
                    + torch.mean(
                        torch.abs(
                            flow_samples_torsions_mean - val_samples_perm_torsions_mean
                        )
                    )
                ).item()
            )

        best_permutation = permutations[errors.index(min(errors))]

        if best_permutation != tuple(hydrogens_cart_indices):
            print(
                f"Cart. validation datasets: Permuting hydrogens attached to {group_carbon} from cart. indices {hydrogens_cart_indices} to {best_permutation}"
            )
            for T in cart_datasets_to_apply_permutation_to.keys():
                current_dataset = cart_datasets_to_apply_permutation_to[T].view(
                    cart_datasets_to_apply_permutation_to[T].shape[0], -1, 3
                )
                current_dataset[:, hydrogens_cart_indices, :] = current_dataset[
                    :, torch.tensor(best_permutation), :
                ]
                cart_datasets_to_apply_permutation_to[T] = current_dataset.view(
                    current_dataset.shape[0], -1
                )

            permutation_applied = True
    return permutation_applied


def get_torsion_indices_for_permutation_constraints(
    system: MolecularSystem,
    reference_minimum_energy_structure_cart: torch.Tensor,
) -> bool:

    reference_minimum_energy_structure_torsions = (
        system.coordinate_trafo_openmm.forward(
            reference_minimum_energy_structure_cart.cuda(), inverse=False
        )[2][0, :]
        .detach()
        .cpu()
    )

    permute_hydrogens_of = []
    hydrogens_of_groups = {}
    for group_center_atom in system.openmm_system.topology.atoms():
        if not (
            group_center_atom.element.name == "carbon"
            or group_center_atom.element.name == "nitrogen"
        ):
            continue

        # Get all hydrogens connected to this center atom:
        hydrogens = [
            (bond[0] if bond[0].name.startswith("H") else bond[1])
            for bond in system.openmm_system.topology.bonds()
            if (
                (bond[0] == group_center_atom and bond[1].name.startswith("H"))
                or (bond[1] == group_center_atom and bond[0].name.startswith("H"))
            )
        ]

        if len(hydrogens) > 1:
            permute_hydrogens_of.append(group_center_atom)
            hydrogens_of_groups[group_center_atom] = hydrogens

    constrain_left = []
    constrain_right = []

    for group_center_atom in permute_hydrogens_of:
        hydrogens = hydrogens_of_groups[group_center_atom]
        hydrogens_cart_indices = [hydrogen.index for hydrogen in hydrogens]

        ##### Determine which IC DOF belong to the atoms in this permutation groups #####

        all_torsions = system.coordinate_trafo_openmm.torsion_indices
        relevant_torsion_indices = []
        for i, torsion in enumerate(all_torsions):
            if any([(index in torsion) for index in hydrogens_cart_indices]):
                relevant_torsion_indices.append(i)

        if len(relevant_torsion_indices) == 3:
            assert all(
                item in hydrogens_cart_indices
                for item in all_torsions[relevant_torsion_indices, :][:, 0]
            )
            assert np.all(
                all_torsions[relevant_torsion_indices, :][1:, 1]
                == all_torsions[relevant_torsion_indices, :][0, 1]
            )
            assert np.all(
                all_torsions[relevant_torsion_indices, :][1:, 2]
                == all_torsions[relevant_torsion_indices, :][0, 2]
            )
            assert np.all(
                all_torsions[relevant_torsion_indices, :][1:, 3]
                == all_torsions[relevant_torsion_indices, :][0, 0]
            )

            indices_to_look_at = [1, 2]
        elif len(relevant_torsion_indices) == 2:
            indices_to_look_at = [0, 1]
        else:
            raise ValueError(
                f"Expected 2 or 3 torsions for a CHX group, got {len(relevant_torsion_indices)}."
            )

        for current_index in indices_to_look_at:
            if (
                reference_minimum_energy_structure_torsions[
                    relevant_torsion_indices[current_index]
                ]
                > 0.5
            ):
                constrain_right.append(relevant_torsion_indices[current_index])
            else:
                constrain_left.append(relevant_torsion_indices[current_index])

    return constrain_right, constrain_left


def apply_permutation(
    val_datasets_IC: dict,
    val_datasets_cart: dict,
    generator: BoltzmannGenerator,
    IC_trafo: SequentialFlow,
    system: MolecularSystem,
    main_temp: float,
):
    permutation_updated = apply_cart_permutation(
        cart_datasets_to_apply_permutation_to=val_datasets_cart,
        generator=generator,
        coordinate_trafo_openmm=system.coordinate_trafo_openmm,
        val_samples_cart=val_datasets_cart[main_temp][0:1000],  # This should be enough.
        system=system.openmm_system,
    )
    if permutation_updated or val_datasets_IC == {}:
        for T in val_datasets_cart.keys():
            val_datasets_IC[T] = convert_to_IC(val_datasets_cart[T], IC_trafo=IC_trafo)
